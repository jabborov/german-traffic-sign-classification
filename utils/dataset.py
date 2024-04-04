from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms

from PIL import Image
import csv
import pathlib

class TrafficSignDataset(Dataset):
    """Loading and reading datasets"""
    def __init__(self, root: str, split: str, transform: bool):      
                
        self.base_folder = pathlib.Path(root)
        self.split = split
        # self.csv_file = None
        if self.split == 'train':
            self.csv_file = self.base_folder / 'Train.csv' 
        else:
            self.csv_file = self.base_folder /'Test.csv'        
        print("***", self.csv_file)
        
        with open(str(self.csv_file)) as csvfile:
            samples = [(str(self.base_folder / row['Path']),int(row['ClassId'])) 
            for row in csv.DictReader(csvfile,delimiter=',',skipinitialspace=True)]

        self.samples = samples
        
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        path, classId =  self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform: 
            sample = self.transform(sample)           
        return sample,classId

def dataAugmentation(split="train"):
    """Apply data augmentation methods for train, validation and test datasets"""  
    
    if split == 'train':
        trainTransforms = transforms.Compose([
            transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
            transforms.AugMix(),
            transforms.RandomEqualize(0.5),
            transforms.GaussianBlur(3, 3),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return trainTransforms
    
    elif split == 'validation':
        validationTransform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])        
        return validationTransform
    
    else:
        testTransform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
        return testTransform   

def train_validation_split(dataset, trainSize: float):
    """Split Training dataset into train and validation datasets"""

    trainSize = int(trainSize * len(dataset)) 
    validationSize = int(len(dataset) - trainSize)
    return random_split(dataset, [trainSize, validationSize])




