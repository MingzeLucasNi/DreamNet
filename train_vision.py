import os
import torch
from torch import nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from dataload import *
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet50
import torchvision
from sleepnet import *
from transformers import ViTModel
# Add this import to the top of your script
import gc

# Add this function to periodically clear GPU memory
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
from torchvision import transforms

# Define your transforms
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to match ResNet50's expected input
])
# mae = ViTModel.from_pretrained("google/vit-large-patch16-224")
mae = ViTModel.from_pretrained("google/vit-base-patch16-224",output_hidden_states=True)

# Freezing the parameters
for param in mae.parameters():
    param.requires_grad = False

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your model
    model = ResNet18(num_classes=args.num_classes).to(device)  # Change as necessary

# Data preprocessing
    transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])
    train_data = get_dataset(args.dataset, train=True, transform=transform)
    val_data = get_dataset(args.dataset, train=False, transform=transform)

    train_l = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_l = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08)

    # Learning Rate Scheduler
    warmup_epochs = int(args.epochs * 0.1)  # 10% of total epochs as warmup
    total_steps = len(train_l) * args.epochs
    warmup_steps = len(train_l) * warmup_epochs

    lr_lambda = lambda step: min(step / warmup_steps, 1) if warmup_steps != 0 else 1
    scheduler = LambdaLR(optimizer, lr_lambda)
  
    
# Training function
    for epoch in range(args.epochs):
        # Training2
        model.train()
        for i, (data, labels) in enumerate(train_l):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            logits= model(data)
            # pixels=mae.unpatchify(pixels)
            # Custom loss: MSE of ViTMAE + Entropy loss eof RsNet
            loss = F.cross_entropy(logits, labels)
            # loss2 = F.mse_loss(data, pixels)
            # loss = loss1 + 0.1*loss2

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


            if (i + 1) % 36 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_l)}], Loss: {loss.item()}')
                clear_memory()
            # print(next(model.parameters()).device)

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss= 0
            for i, (data, labels) in enumerate(val_l):
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_1=F.cross_entropy(logits, labels)
                # pixels=mae.unpatchify(pixels)
                loss += loss_1
            print(f'Validation loss: {loss.item()}')


            print(f'Validation accuracy: {100 * correct / total}%')

        # Save the model after each epoch
        if (epoch + 1) % 2 == 0:
            folder_path=args.dataset+'_resnet50'
            path = f'./{folder_path}/'
            # Make the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)

            # Save the model
            torch.save(model.state_dict(), os.path.join(path, f'_model_epoch_{epoch+1}.pth'))
            # torch.save(model.state_dict(), f'{path}_model_epoch_{epoch+1}.pth')
            # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        # Reset scheduler for the next epoch
        scheduler = LambdaLR(optimizer, lr_lambda)
def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="cifar10", choices=dataset_choices.keys())
    parser.add_argument("--dataset", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'imagenet'])
    # parser.add_argument("--dataset", type=class, default=cifar10) # Set type to str and map it to Dataset in your function
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    # args.dataset = dataset_choices[args.dataset]
    return args



if __name__ == '__main__':
    args = args_parser()
    print(args)
    train_model(args)

# Call the training function
# train_model(model, optimizer, train_loader, val_loader, num_epochs, device)

# nohup python -u train_ResNet50.py > cifar10_Resnet50.txt 2>&1 &
# nohup python -u train_ResNet50.py --dataset 'cifar100' --epochs 50 --num_classes 100 > cifar100_Resnet18.txt 2>&1 &
