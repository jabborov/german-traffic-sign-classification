import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import TrafficSignModel
from utils.dataset import TrafficSignDataset, dataAugmentation, train_validation_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, device, opt):
    """Train the model and keep the weights with the best validation accuracy."""
    model.to(device)

    save_dir = pathlib.Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_weight = save_dir / "best.pt"

    # Optimizer, learning rate scheduler, loss function
    optimizer = Adam(params=model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=opt.lr_patience)
    criterion = nn.CrossEntropyLoss()

    # Datasets: each split gets its own transform
    dataset = TrafficSignDataset(root=opt.path, split="train", transform=None)
    train_data, validation_data = train_validation_split(
        dataset, trainSize=0.8, seed=opt.seed,
        train_transform=dataAugmentation("train"),
        validation_transform=dataAugmentation("validation"),
    )
    print(f"Train set size: {len(train_data)}, Validation set size: {len(validation_data)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_data, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.workers, pin_memory=pin_memory)

    best_score = 0.0
    epochs_without_improvement = 0

    for epoch in range(opt.epochs):
        model.train()  # validate() switches to eval mode, so set train mode every epoch

        running_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        for image, label in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}"):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label.size(0)
            predicted = output.argmax(1)
            correct_predictions += (predicted == label).sum().item()
            total_examples += label.size(0)

        train_loss = running_loss / total_examples
        train_acc = correct_predictions / total_examples
        print(f"Epoch {epoch}, Train Loss [{train_loss:.4f}], Train Accuracy [{train_acc:.4f}]")

        val_acc, val_loss = validate(model, validation_loader, device, criterion)
        print(f"Validation Loss [{val_loss:.4f}], Validation Accuracy [{val_acc:.4f}], "
              f"LR [{optimizer.param_groups[0]['lr']:.2e}]")

        scheduler.step(val_acc)

        if val_acc > best_score:
            best_score = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_weight)
            print(f"Saved new best weights to {best_weight}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opt.patience:
                print(f"Early stopping: no improvement for {opt.patience} epochs.")
                break

    print(f"Best validation accuracy: {best_score:.4f}")


def validate(model, dataloader, device, criterion):
    """Evaluate on the validation dataset. Returns (accuracy, mean loss)."""
    model.eval()

    running_loss = 0.0
    true_outputs, total_samples = 0, 0

    with torch.no_grad():
        for image, label in tqdm(dataloader, total=len(dataloader), desc="Validation"):
            image, label = image.to(device), label.to(device)
            output = model(image)
            running_loss += criterion(output, label).item() * label.size(0)
            true_outputs += (output.argmax(1) == label).sum().item()
            total_samples += label.size(0)

    return true_outputs / total_samples, running_loss / total_samples


def parse_opt():
    parser = argparse.ArgumentParser(description="Traffic Sign Classification training parameters")
    parser.add_argument("--path", type=str, default="./gtrsb_dataset", help="Path to base folder of dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping: epochs without validation improvement")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Epochs without improvement before the learning rate is halved")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="./weights", help="Path to save weights")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (also fixes the train/val split)")

    return parser.parse_args()


def main(opt):
    set_seed(opt.seed)
    model = TrafficSignModel(opt.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, device, opt)


if __name__ == "__main__":
    main(parse_opt())
