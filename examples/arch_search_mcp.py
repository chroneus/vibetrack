#!/usr/bin/env python3
"""Architecture search using vibetrack MCP + Claude Agent SDK.

Trains 5 tiny MLP variants on a sine-regression task (numpy, no GPU needed),
logs metrics to vibetrack, starts the MCP server, then asks Claude to query
the server and recommend the best architecture.

No API key required — uses the local Claude Code CLI (Claude Pro subscription).

Runtime: ~2-3 minutes.

Requirements:
    pip install vibetrack[mcp] numpy claude-agent-sdk

Usage:
    python examples/arch_search_mcp.py
"""

from __future__ import annotations

import anyio
import subprocess
import sys
import time

import numpy as np

import vibetrack

PROJECT_FOLDER = "/tmp/vibetrack_arch_search"
MCP_PORT = 16006

# ── Architecture candidates ───────────────────────────────────────────────────

ARCHS = [
    {"name": "tiny",   "layers": [16],          "lr": 0.05},
    {"name": "small",  "layers": [32, 16],       "lr": 0.01},
    {"name": "medium", "layers": [64, 32],       "lr": 0.01},
    {"name": "deep",   "layers": [32, 32, 32],   "lr": 0.005},
    {"name": "wide",   "layers": [128, 64],      "lr": 0.001},
]

# ── Tiny numpy MLP ────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def train(cfg: dict, epochs: int = 300) -> list[tuple[int, float, float]]:
    """Train a tiny MLP on y = sin(x) with SGD; return (epoch, train_loss, val_loss) tuples."""
    np.random.seed(42)

    X_tr = np.linspace(0, 2 * np.pi, 120).reshape(-1, 1)
    y_tr = np.sin(X_tr)
    X_va = np.linspace(0.05, 2 * np.pi + 0.05, 40).reshape(-1, 1)
    y_va = np.sin(X_va)

    layer_sizes = [1] + cfg["layers"] + [1]
    Ws = [np.random.randn(a, b) * np.sqrt(2.0 / a) for a, b in zip(layer_sizes, layer_sizes[1:])]
    bs = [np.zeros((1, b)) for b in layer_sizes[1:]]

    lr = cfg["lr"]
    history: list[tuple[int, float, float]] = []

    def forward(X: np.ndarray):
        acts = [X]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(Ws) - 1 else relu(z))
        return acts

    for epoch in range(epochs):
        acts = forward(X_tr)
        diff = acts[-1] - y_tr
        train_loss = float(np.mean(diff ** 2))

        delta = 2 * diff / len(y_tr)
        for i in range(len(Ws) - 1, -1, -1):
            dW = acts[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ Ws[i].T) * relu_grad(acts[i])
            Ws[i] -= lr * dW
            bs[i] -= lr * db

        if epoch % 10 == 0:
            va_acts = forward(X_va)
            val_loss = float(np.mean((va_acts[-1] - y_va) ** 2))
            history.append((epoch, train_loss, val_loss))

    return history

# ── Step 1: run experiments ───────────────────────────────────────────────────

def run_experiments() -> None:
    print("=== Step 1: Training architectures ===")
    for cfg in ARCHS:
        n_params = sum(a * b for a, b in zip([1] + cfg["layers"], cfg["layers"] + [1]))
        print(f"  {cfg['name']:8s}  layers={cfg['layers']}  lr={cfg['lr']}  params={n_params}", end="", flush=True)

        history = train(cfg)

        writer = vibetrack.SummaryWriter(
            log_dir=f"{PROJECT_FOLDER}/{cfg['name']}",
            name=cfg["name"],
            project_folder=PROJECT_FOLDER,
            config={"layers": str(cfg["layers"]), "lr": cfg["lr"], "n_params": n_params},
        )
        for epoch, train_loss, val_loss in history:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.close()

        final_val = history[-1][2]
        print(f"  →  val_loss={final_val:.5f}")

    print(f"\n  Logs saved to {PROJECT_FOLDER}\n")

# ── Step 2: start the MCP server ──────────────────────────────────────────────

def start_mcp_server() -> subprocess.Popen:
    print(f"=== Step 2: Starting vibetrack MCP server (port {MCP_PORT}) ===")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vibetrack.viewers.mcp",
         "--project-folder", PROJECT_FOLDER,
         "--port",   str(MCP_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    print(f"  Ready at http://127.0.0.1:{MCP_PORT}/mcp\n")
    return proc

# ── Step 3: ask Claude via MCP ────────────────────────────────────────────────

PROMPT = """\
You are an ML engineer reviewing architecture search results logged to vibetrack.

Use the available MCP tools to:
1. List all experiments.
2. Fetch the summary table (final metrics per experiment).
3. For each experiment, get the "loss/val" scalar series.
4. Identify which architecture converges fastest and which achieves the lowest
   final validation loss.
5. Give a clear recommendation (one best overall architecture) with concise reasoning.

Focus on val_loss. Keep your final answer under 200 words.
"""

async def ask_claude() -> None:
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    print("=== Step 3: Claude querying MCP server ===\n")

    async for message in query(
        prompt=PROMPT,
        options=ClaudeAgentOptions(
            mcp_servers={
                "vibetrack": {
                    "type": "http",
                    "url": f"http://127.0.0.1:{MCP_PORT}/mcp",
                }
            },
            permission_mode="default",
            max_turns=20,
        ),
    ):
        if isinstance(message, ResultMessage):
            print(message.result)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    run_experiments()

    proc = start_mcp_server()
    try:
        anyio.run(ask_claude)
    finally:
        proc.terminate()
        print("\n=== Done ===")

if __name__ == "__main__":
    main()
