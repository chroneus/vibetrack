"""MNIST training with vibetrack W&B-style API.

Run:  python examples/mnist_wandb.py
View: vibetrack  then open /<current-project>
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import vibetrack as wandb


run = wandb.init(
    project="mnist",
    config={"lr": 1e-3, "batch_size": 64, "epochs": 5, "optimizer": "AdamW"},
)


# 2. Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Model
model = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. Train
for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            acc = (output.argmax(1) == target).float().mean().item()
            run.log({"train/loss": loss.item(), "train/acc": acc})

            # Log sample images with predictions
            with torch.no_grad():
                preds = output[:8].argmax(dim=1)
                grid = make_grid(data[:8].cpu(), nrow=4, normalize=True)
                caption = "preds: " + " ".join(str(p.item()) for p in preds)
                run.log(
                    {
                        "train/samples": wandb.Image(grid),
                        "train/predictions": caption,
                    }
                )

    print(f"Epoch {epoch+1}/5 done")


print("\nView results:")
print("  vibetrack")
