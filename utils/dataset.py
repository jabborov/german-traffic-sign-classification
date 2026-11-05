import csv
import pathlib
from typing import Callable, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

# Preprocessing constants. inference.cc must use the same values.
IMAGE_SIZE = 32
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class TrafficSignDataset(Dataset):
    """Reads GTSRB samples listed in Train.csv / Test.csv (columns: ClassId, Path)."""

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.base_folder = pathlib.Path(root)
        self.split = split
        self.csv_file = self.base_folder / ("Train.csv" if split == "train" else "Test.csv")
        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        with open(self.csv_file, newline="") as csvfile:
            self.samples = [
                (str(self.base_folder / row["Path"]), int(row["ClassId"]))
                for row in csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
            ]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_id = self.samples[index]
        with Image.open(path) as img:
            sample = img.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, class_id


class TransformedSubset(Dataset):
    """A subset with its own transform, so train and validation splits do not share one."""

    def __init__(self, subset: Subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        sample, class_id = self.subset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, class_id


def dataAugmentation(split: str = "train"):
    """Transforms for the train, validation and test datasets.

    Flips are not used: they change the meaning of direction signs
    (e.g. classes 33/34 turn right/left, 36/37, 38/39 keep right/left).
    """
    base = [
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    ]
    tail = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]

    if split == "train":
        augment = [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
        ]
        return transforms.Compose(base + augment + tail)

    if split in ("validation", "test"):
        return transforms.Compose(base + tail)

    raise ValueError(f"Unknown split: {split!r}")


def train_validation_split(dataset: Dataset, trainSize: float, seed: int = 0,
                           train_transform: Optional[Callable] = None,
                           validation_transform: Optional[Callable] = None):
    """Split the training dataset into train and validation datasets.

    `dataset` should be created with transform=None; each split gets its own transform.
    A fixed seed makes the split reproducible.
    """
    train_len = int(trainSize * len(dataset))
    validation_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = random_split(dataset, [train_len, validation_len], generator=generator)
    return (TransformedSubset(train_subset, train_transform),
            TransformedSubset(validation_subset, validation_transform))
