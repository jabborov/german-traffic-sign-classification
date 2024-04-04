from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from model.model import TrafficSignModel
from utils.dataset import TrafficSignDataset, dataAugmentation

def  evaluate(model, device, opt):
    """Evaluating the test dataset"""
    
    correct_predictions = 0
    total_examples = 0

    # Move the model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Test dataset
    dataset = TrafficSignDataset(root=opt.path, split="test", transform=True) 
    print(f"Test set size:  {len(dataset)}") 

    # Data Augmentation
    dataset.transform = dataAugmentation('test')

    # Data Loader
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    # Iterate over the test dataset
    for image, label in tqdm(test_loader, total=len(test_loader)):
        image, label = image.to(device), label.to(device)        

        output= model(image)       

        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_examples += label.size(0)        
    print(f'Test Accuracy [{"{:.4f}".format(correct_predictions/total_examples)}]')  

def parse_opt():
    parser = argparse.ArgumentParser(description="Traffic Sign Classification evaluation parameters")
    parser.add_argument("--path", type=str, default="./gtrsb_dataset", help="Path to base folder of dataset")    
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--weights", type=str, default='./weights/best.pt', help="Load pretrained pytorch weight file")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")
 
    return parser.parse_args()

def main(opt):
    # Call to Model
    model = TrafficSignModel(opt.classes)

    # Load weight files
    state_dict = torch.load(opt.weights)
    model.load_state_dict(state_dict)    
    
    # Determine device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(model, device, opt)

if __name__ == "__main__":
    parameters = parse_opt()
    main(parameters)