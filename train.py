from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
import torch.nn.functional as F

from model.model import TrafficSignModel
from utils.dataset import TrafficSignDataset, dataAugmentation, train_validation_split

def train(model, device, opt):
    """ Training the train dataset """
    start_epoch = 0
    best_score = 0

    # Move the model to the specified device
    model.to(device)

    # Set the model to train mode
    model.train()

    # Optimizer, Learning Rate Scheduler, Loss function
    optimizer = Adam(params=model.parameters(),lr=opt.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)
    criterion = nn.CrossEntropyLoss()  

    # Datasets
    dataset = TrafficSignDataset(root=opt.path, split="train", transform=True)   
    train_data, validation_data = train_validation_split(dataset, trainSize=0.8)
    print(f"Train set size:  {len(train_data)}, Validation set size: , {len(validation_data)}")

    # Data Augmentation
    train_data.dataset.transform = dataAugmentation('train')
    validation_data.dataset.transform = dataAugmentation('validation')

    # Data Loader
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=opt.batch_size)  
    
    # Training
    for epoch in range(start_epoch, opt.epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_examples = 0
        best_weight = opt.save_dir + "/best.pt"        
        
        # Iterate over the train dataset
        for image, label in tqdm(train_loader, total=len(train_loader)):
            image, label = image.to(device), label.to(device)

            output= model(image)
            loss = criterion(output, label)  

            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            total_examples += label.size(0)
            
        print(f'Epoch {epoch}, Train Loss [{"{:.4f}".format(epoch_loss)}], Train Accuracy [{"{:.4f}".format(correct_predictions/total_examples)}]')

        val_acc, val_loss = validate(model, validation_loader, device, criterion)
        print(f'Validation Loss [{"{:.4f}".format(val_loss)}], Validation Accuracy [{"{:.4f}".format(val_acc)}]')

        scheduler.step()
        
        if best_score < val_acc:
            best_score = max(best_score, val_acc)
            torch.save(model.state_dict(), best_weight)           

def validate(model, dataloader, device, criterion):
    """ Evaluating the validation dataset """

    true_outputs, total_samples = 0, 0 
    
    # Set the model to evaluation mode
    model.eval()   

    # Iterate over the validation dataset
    for image, label in tqdm(dataloader, total=len(dataloader)):
        image, label = image.to(device), label.to(device)
        
        with torch.no_grad():
            output = model(image)             
            loss = criterion(output,label)
            val_loss = loss.item()
            _, output = torch.max(output,1)
            true_outputs += (output == label).sum().item()
            total_samples += label.size(0)

    val_acc = true_outputs / total_samples

    return val_acc, val_loss

def parse_opt():
    parser = argparse.ArgumentParser(description="Traffic Sign Classification training parameters")
    parser.add_argument("--path", type=str, default="./gtrsb_dataset", help="Path to base folder of dataset")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch sizes")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="./weights", help="Path to save weights")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")

    return parser.parse_args()

def main(opt):
    # Call the model
    model = TrafficSignModel(opt.classes)

    # Determine device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the train() 
    train(model, device, opt)

if __name__ == "__main__":
    parameters = parse_opt()
    main(parameters)
