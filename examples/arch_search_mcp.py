#!/usr/bin/env python3
"""Architecture-search demo using vibetrack MCP plus any tool-calling LLM.

The script trains a few tiny numpy MLP variants on a sine-regression task,
logs their metrics to vibetrack, starts the vibetrack MCP server, exposes the
MCP tools to an OpenAI-compatible chat-completions endpoint, and asks the LLM
to inspect the experiment data before recommending a model.

Works with local or hosted OpenAI-compatible APIs that support tool calling:
Ollama, LM Studio, vLLM, OpenAI, etc.

Requirements:
    pip install vibetrack[all] numpy httpx

Example with Ollama:
    ollama pull qwen3-coder:30b
    LLM_BASE_URL=http://127.0.0.1:11434/v1 \
    LLM_API_KEY=ollama \
    LLM_MODEL=qwen3-coder:30b \
    python examples/arch_search_mcp.py


Example with OpenAI:
    LLM_BASE_URL=https://api.openai.com/v1 \
    LLM_API_KEY=$OPENAI_API_KEY \
    LLM_MODEL=gpt-4.1-mini \
    python examples/arch_search_mcp.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np

import vibetrack

PROJECT_FOLDER = Path(os.getenv("VT_MCP_DEMO_PROJECT", "/tmp/vibetrack_mcp_llm_demo"))
MCP_PORT = int(os.getenv("VT_MCP_DEMO_MCP_PORT", "16006"))
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/mcp"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-coder:30b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TURNS = int(os.getenv("LLM_MAX_TURNS", "12"))
LLM_UNLOAD_ON_EXIT = os.getenv("LLM_UNLOAD_ON_EXIT", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}


ARCHITECTURES = [
    {"name": "tiny_fast", "layers": [16], "lr": 0.045},
    {"name": "small_balanced", "layers": [32, 16], "lr": 0.012},
    {"name": "medium_stable", "layers": [64, 32], "lr": 0.008},
    {"name": "deep_slow", "layers": [32, 32, 32], "lr": 0.004},
    {"name": "wide_underfit", "layers": [128, 64], "lr": 0.001},
]


PROMPT = """\
You are reviewing a vibetrack architecture search.

Use MCP tools before answering:
1. list_experiments
2. compare_scalar for loss/val with objective="min"
3. analyze_scalar for the strongest candidates
4. get_hparams for the winner and one close alternative

Decide which run is best for production. Focus on validation loss, convergence
speed, and parameter count. Mention the evidence you used. Keep the answer
under 220 words.
"""

REQUIRED_MCP_TOOLS = (
    "list_experiments",
    "compare_scalar",
    "analyze_scalar",
    "get_hparams",
)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def train(cfg: Dict[str, Any], epochs: int = 260) -> List[tuple[int, float, float]]:
    """Train a tiny MLP on y = sin(x); return (epoch, train_loss, val_loss)."""
    rng = np.random.default_rng(7)

    x_train = np.linspace(0, 2 * np.pi, 128).reshape(-1, 1)
    y_train = np.sin(x_train)
    x_val = np.linspace(0.04, 2 * np.pi + 0.04, 48).reshape(-1, 1)
    y_val = np.sin(x_val)

    layer_sizes = [1] + list(cfg["layers"]) + [1]
    weights = [
        rng.normal(0, np.sqrt(2.0 / a), size=(a, b))
        for a, b in zip(layer_sizes, layer_sizes[1:])
    ]
    biases = [np.zeros((1, b)) for b in layer_sizes[1:]]

    def forward(x: np.ndarray) -> List[np.ndarray]:
        acts = [x]
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = acts[-1] @ w + b
            acts.append(z if i == len(weights) - 1 else relu(z))
        return acts

    history: List[tuple[int, float, float]] = []
    for epoch in range(epochs + 1):
        acts = forward(x_train)
        diff = acts[-1] - y_train
        train_loss = float(np.mean(diff**2))

        delta = 2 * diff / len(y_train)
        for i in range(len(weights) - 1, -1, -1):
            dw = acts[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ weights[i].T) * relu_grad(acts[i])
            weights[i] -= cfg["lr"] * dw
            biases[i] -= cfg["lr"] * db

        if epoch % 20 == 0:
            val_pred = forward(x_val)[-1]
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            history.append((epoch, train_loss, val_loss))

    return history


def parameter_count(layers: Iterable[int]) -> int:
    sizes = [1] + list(layers) + [1]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def validation_scorecard_text() -> str:
    """Compact deterministic fallback context for models that over-call tools."""
    rows = []
    for cfg in ARCHITECTURES:
        history = train(cfg)
        best_epoch, _best_train, best_val = min(history, key=lambda row: row[2])
        final_epoch, _final_train, final_val = history[-1]
        rows.append(
            {
                "name": cfg["name"],
                "params": parameter_count(cfg["layers"]),
                "best_epoch": best_epoch,
                "best_val": best_val,
                "final_epoch": final_epoch,
                "final_val": final_val,
            }
        )

    rows.sort(key=lambda row: row["final_val"])
    lines = ["Validation scorecard; lower loss/val is better:"]
    for row in rows:
        lines.append(
            f"- {row['name']}: final={row['final_val']:.5f} "
            f"at epoch {row['final_epoch']}, best={row['best_val']:.5f} "
            f"at epoch {row['best_epoch']}, params={row['params']}"
        )
    lines.append(f"Lowest final loss/val: {rows[0]['name']}.")
    return "\n".join(lines)


def seed_experiments() -> None:
    """Create a fresh project with enough signal for a meaningful LLM answer."""
    if PROJECT_FOLDER.exists():
        shutil.rmtree(PROJECT_FOLDER)
    PROJECT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("1. Training and logging architecture candidates")
    for cfg in ARCHITECTURES:
        params = parameter_count(cfg["layers"])
        history = train(cfg)
        final_val = history[-1][2]
        best_val = min(row[2] for row in history)

        print(
            f"   {cfg['name']:<15} layers={cfg['layers']} "
            f"lr={cfg['lr']:<7} params={params:<5} final_val={final_val:.5f}"
        )

        with vibetrack.SummaryWriter(
            log_dir=str(PROJECT_FOLDER / cfg["name"]),
            name=cfg["name"],
            project_folder=str(PROJECT_FOLDER),
            system_metrics_interval=0,
            config={
                "layers": str(cfg["layers"]),
                "lr": cfg["lr"],
                "params": params,
            },
        ) as writer:
            writer.add_text(
                "notes",
                (
                    f"Candidate {cfg['name']} has layers={cfg['layers']}, "
                    f"lr={cfg['lr']}, params={params}, best_val={best_val:.6f}."
                ),
                global_step=0,
            )
            for epoch, train_loss, val_loss in history:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)

    print(f"   wrote project: {PROJECT_FOLDER}\n")


def wait_for_mcp(proc: subprocess.Popen[Any], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "MCP server exited before it was reachable. "
                "Install MCP support with `pip install -e .[all]` or "
                "`pip install vibetrack[all]`."
            )
        try:
            with socket.create_connection(("127.0.0.1", MCP_PORT), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"MCP server did not become reachable at {MCP_URL}")


def start_mcp_server() -> subprocess.Popen[Any]:
    print("2. Starting vibetrack MCP server")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vibetrack.viewers.mcp",
            "--project-folder",
            str(PROJECT_FOLDER),
            "--port",
            str(MCP_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_mcp(proc)
    print(f"   MCP endpoint: {MCP_URL}\n")
    return proc


def tool_result_text(result: Any) -> str:
    parts: List[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        elif hasattr(item, "model_dump"):
            parts.append(json.dumps(item.model_dump(), default=str))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def ollama_base_url() -> Optional[str]:
    parsed = urlsplit(LLM_BASE_URL)
    if not parsed.scheme or not parsed.netloc:
        return None
    hostname = parsed.hostname or ""
    is_ollama = LLM_API_KEY == "ollama" or parsed.port == 11434 or "ollama" in hostname
    if not is_ollama:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def unload_ollama_model() -> None:
    if not LLM_UNLOAD_ON_EXIT:
        return
    base_url = ollama_base_url()
    if not base_url:
        return

    print(f"4. Unloading Ollama model: {LLM_MODEL}")
    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json={"model": LLM_MODEL, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"   warning: could not unload Ollama model: {exc}")
    else:
        print("   unloaded\n")


async def list_mcp_tools(session: Any) -> List[Dict[str, Any]]:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return tools


async def chat_completion(
    client: httpx.AsyncClient,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }
    if tools:
        payload.update({"tools": tools, "tool_choice": "auto"})

    response = await client.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def ask_llm_via_mcp() -> None:
    """Expose vibetrack MCP tools to any OpenAI-compatible tool-calling LLM."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    print("3. Asking LLM to query vibetrack through MCP")
    print(f"   LLM endpoint: {LLM_BASE_URL}")
    print(f"   LLM model:    {LLM_MODEL}\n")

    async with streamable_http_client(MCP_URL) as (read, write, _session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_mcp_tools(session)
            print("   MCP tools exposed to LLM:")
            for tool in tools:
                print(f"   - {tool['function']['name']}")
            print()

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful ML experiment analyst. Use tools "
                        "for evidence; do not invent experiment results."
                    ),
                },
                {"role": "user", "content": PROMPT},
            ]

            async with httpx.AsyncClient() as client:
                used_tools: List[str] = []
                for _ in range(LLM_MAX_TURNS):
                    message = await chat_completion(client, messages, tools)
                    messages.append(message)
                    tool_calls = message.get("tool_calls") or []
                    missing_tools = [
                        name for name in REQUIRED_MCP_TOOLS if name not in used_tools
                    ]

                    if not tool_calls:
                        content = (message.get("content") or "").strip()
                        if missing_tools:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Continue with the required MCP tool calls before "
                                        f"the final recommendation. Still missing: "
                                        f"{', '.join(missing_tools)}."
                                    ),
                                }
                            )
                            continue
                        if content:
                            print("=== LLM recommendation ===")
                            print(content)
                            return
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was empty. Provide the final "
                                    "recommendation in plain text using the tool results."
                                ),
                            }
                        )
                        continue

                    for call in tool_calls:
                        name = call["function"]["name"]
                        used_tools.append(name)
                        raw_args = call["function"].get("arguments") or {}
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = raw_args
                        print(f"   tool call: {name}({json.dumps(args)})")
                        result = await session.call_tool(name, args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": tool_result_text(result),
                            }
                        )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Stop calling tools now. Based only on the MCP tool "
                            "results already in this conversation, give the final "
                            "production recommendation in under 220 words.\n\n"
                            f"{validation_scorecard_text()}"
                        ),
                    }
                )
                message = await chat_completion(client, messages)
                content = (message.get("content") or "").strip()
                if content:
                    print("=== LLM recommendation ===")
                    print(content)
                    return

            raise RuntimeError("LLM did not provide a final recommendation")


def main() -> None:
    seed_experiments()
    proc = start_mcp_server()
    try:
        asyncio.run(ask_llm_via_mcp())
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        unload_ollama_model()
        print("\nDone.")


if __name__ == "__main__":
    main()
