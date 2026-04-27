from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
from vibetrack import SummaryWriter

# 1. Load Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Define Simple Model
model = nn.Sequential(
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
)
writer = SummaryWriter("mnist/adamw_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# 3. Setup Optimizer and Loss
optimizer = optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # --- The Core Loop Steps ---
        optimizer.zero_grad()  # 1. Clear previous gradients
        output = model(data)  # 2. Forward pass
        loss = criterion(output, target)  # 3. Compute loss
        loss.backward()  # 4. Backward pass (calculate gradients)
        optimizer.step()  # 5. Update weights
        # ---------------------------

        global_step = batch_idx + len(train_loader) * epoch
        if batch_idx % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)

            # Log a grid of input images with predicted labels
            with torch.no_grad():
                preds = output[:8].argmax(dim=1)
                imgs = data[:8].cpu()
                # Make a grid: torchvision.utils.make_grid returns CHW tensor
                from torchvision.utils import make_grid

                grid = make_grid(imgs, nrow=4, normalize=True)
                caption = "preds: " + " ".join(str(p.item()) for p in preds)
                writer.add_image("train/samples", grid, global_step)
                writer.add_text("train/predictions", caption, global_step)


# Run training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in tqdm.tqdm(range(5)):
    train(model, device, train_loader, optimizer, criterion, epoch)
writer.close()

print("\nView results:")
print("  vibetrack")
