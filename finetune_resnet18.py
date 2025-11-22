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
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class Config:
    seed: int = 42
    data_dir: str = "data_resnet_finetune"
    log_dir: str = "runs_resnet_finetune"
    ckpt_dir: str = "checkpoints_resnet_finetune"

    batch_size: int = 128
    num_workers: int = 2
    num_epochs: int = 25
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4

    val_ratio: float = 0.2


Config = Config()


# -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(Config.seed)

##------Dataset and dataloader---------
# cifar10 mean and std
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms() #only resize to 224 and normalize image

train_set = torchvision.datasets.CIFAR10(root=Config.data_dir, train=True, download=True, transform=preprocess)
test_set = torchvision.datasets.CIFAR10(root=Config.data_dir, train=False, download=True, transform=preprocess)

# split train/val
val_size = int(len(train_set) * Config.val_ratio)
train_size = len(train_set) - val_size
train_set, val_set = random_split(train_set, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(Config.seed))

train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)


##--------Resnet18--------------
class ResNet18_finetune(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # Load pretrained ResNet18 model
        self.model = resnet18(weights=weights)

        #(1) change fc to 10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        #(2) freeze all the backbone
        for p in self.model.parameters():
            p.requires_grad = False

        #unfreeze layer4 and fc only
        for p in self.model.layer4.parameters():
            p.requires_grad = True
        for p in self.model.fc.parameters():
            p.requires_grad = True

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


# ----------main------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = ResNet18_finetune(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.SGD(
        trainable_params,
        lr=Config.lr,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    os.makedirs(Config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)
    best_val_acc = 0.0
    global_step = 0

    patience = 10
    epochs_no_improve = 0

    for epoch in range(Config.num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer,
                                                             epoch, global_step)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("Acc/train_epoch", train_acc, epoch)
        writer.add_scalar("Acc/val_epoch", val_acc, epoch)
        writer.add_scalar("LR/epoch", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch [{epoch + 1}/{Config.num_epochs}]"
            f"train_loss = {train_loss:.4f} | train_acc = {train_acc:.4f} |"
            f"val_loss = {val_loss:.4f} | val_acc = {val_acc:.4f} |"
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # save last
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

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

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
            print(f"New best val_acc: {best_val_acc:.4f}, saving best.pth")
        else:
            epochs_no_improve += 1
            print(f"No imporvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} with val_acc {val_acc:.4f}")
                break
        writer.close()

    # test
    best_ckpt = torch.load(os.path.join(Config.ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Test loss = {test_loss:.4f} | Test acc = {test_acc:.4f}")


if __name__ == "__main__":
    main()