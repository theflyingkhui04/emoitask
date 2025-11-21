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
from torchvision.models import resnet18

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "data_resnet"
    log_dir: str = "runs_resnet"
    ckpt_dir: str = "checkpoints_resnet"

    batch_size: int = 128
    num_workers: int = 2
    num_epochs: int = 60
    lr: float = 1e-2
    momentum: float = 0.9
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

##--------Resnet18--------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        # Load pretrained ResNet18 model
        self.model = resnet18(weights=None)
        
        # Modify first conv layer for CIFAR-10 (smaller images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool layer for CIFAR-10
        self.model.maxpool = nn.Identity()
        
        # Change final fully connected layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

##----------train/eval------------
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

#----------main------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.lr,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    os.makedirs(Config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(Config.num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, global_step)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("Acc/train_epoch", train_acc, epoch)
        writer.add_scalar("Acc/val_epoch", val_acc, epoch)
        writer.add_scalar("LR/epoch", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch [{epoch+1}/{Config.num_epochs}]"
            f"train_loss = {train_loss:.4f} | train_acc = {train_acc:.4f} |"
            f"val_loss = {val_loss:.4f} | val_acc = {val_acc:.4f} |"
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        #save last
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            os.path.join(Config.ckpt_dir, "last.pth")
        )

        #save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc
                },
                os.path.join(Config.ckpt_dir, "best.pth")
            )

        writer.close()

        #test
        best_ckpt = torch.load(os.path.join(Config.ckpt_dir, "best.pth"), map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        print(f"Test loss = {test_loss:.4f} | Test acc = {test_acc:.4f}")

    if __name__ == "__main__":
        main()