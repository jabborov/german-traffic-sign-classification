import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import TrafficSignModel
from utils.dataset import TrafficSignDataset, dataAugmentation


def evaluate(model, device, opt):
    """Evaluate on the test dataset."""
    correct_predictions = 0
    total_examples = 0

    model.to(device)
    model.eval()

    dataset = TrafficSignDataset(root=opt.path, split="test", transform=dataAugmentation("test"))
    print(f"Test set size: {len(dataset)}")

    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                             num_workers=opt.workers, pin_memory=device.type == "cuda")

    with torch.no_grad():
        for image, label in tqdm(test_loader, total=len(test_loader)):
            image, label = image.to(device), label.to(device)
            predicted = model(image).argmax(1)
            correct_predictions += (predicted == label).sum().item()
            total_examples += label.size(0)

    accuracy = correct_predictions / total_examples
    print(f"Test Accuracy [{accuracy:.4f}]")
    return accuracy


def parse_opt():
    parser = argparse.ArgumentParser(description="Traffic Sign Classification evaluation parameters")
    parser.add_argument("--path", type=str, default="./gtrsb_dataset", help="Path to base folder of dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--weights", type=str, default="./weights/best.pt", help="Pretrained PyTorch weight file")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")

    return parser.parse_args()


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficSignModel(opt.classes)
    # map_location lets GPU-trained weights load on CPU-only machines
    state_dict = torch.load(opt.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    evaluate(model, device, opt)


if __name__ == "__main__":
    main(parse_opt())
