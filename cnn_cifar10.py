import os, random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "data"
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"

    batch_size: int = 128
    num_workers: int = 2
    num_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 5e-4

    val_ratio: float = 0.2

Config = Config()

#-------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(Config.seed)

##------Dataset and dataloader---------
# cifar10 mean and std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_set = torchvision.datasets.CIFAR10(root=Config.data_dir, train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root=Config.data_dir, train=False, download=True, transform=test_transform)

#split train/val
val_size = int(len(train_set) * Config.val_ratio)
train_size = len(train_set) - val_size
train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(Config.seed))

train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)
test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

##--------CNN model-------------
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 64x16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2), #32->16
            nn.Dropout(p=0.1),

            #Block 2: 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2), #16->8
            nn.Dropout(p=0.2),

            #Block 3: 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(p=0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

##-------train/eval function----------
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device, writer, epoch, global_step):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, labels)

        running_loss += loss.item()
        running_acc += acc

        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        writer.add_scalar("Accuracy/train_step", acc, global_step)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

        global_step += 1
    return running_loss / len(loader), running_acc / len(loader), global_step

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(loader), total_acc / len(loader)

def save_checkpoint(state, path):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, "checkpoint.pth")
    torch.save(state, filename)

##--------main train---------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    os.makedirs(Config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(Config.num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, global_step)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{Config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss
            },
            os.path.join(Config.ckpt_dir, "last.pth")
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            os.path.join(Config.ckpt_dir, "best.pth")
        )
    writer.close()

    #test final with best checkpoint
    best_ckpt = torch.load(os.path.join(Config.ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()