import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from argparsers import *
from torchvision import transforms
from utilities import *
from dreamnet import *
#write a logger to record the print statements
def logger(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
        f.write('\n')

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')
    logger_name = args.dataset+'_DreamNet_logger.txt'
    
# Load your model
    # model = args.model.to(device)
    model=ResNet18(args.num_classes).to(device)
    transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])

# Load your dataset
    train_data = get_dataset(args.dataset, train=True, transform=transform)
    val_data = get_dataset(args.dataset, train=False, transform=transform)
    train_l = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_l = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08)
    # Learning Rate Scheduler
    warmup_epochs = int(args.epochs * 0.1) # 10% of total epochs as warmup
    total_steps = len(train_l) * args.epochs
    warmup_steps = len(train_l) * warmup_epochs

    lr_lambda = lambda step: min(step / warmup_steps, 1) if warmup_steps != 0 else 1
    scheduler = LambdaLR(optimizer, lr_lambda)    
# Training 
    for epoch in range(args.epochs):
        model.train()
        for i, (data, labels) in enumerate(train_l):
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()



            if (i + 1) % 36 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_l)}], Loss: {loss.item()}')
                logger(logger_name, f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_l)}], Loss: {loss.item()}')



        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss1 = 0
            loss2 = 0
            loss = 0
            for i, (data, labels) in enumerate(val_l):
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # loss += F.cross_entropy(logits, labels)
            # print(f'Validation loss: {loss.item()}')


            print(f'Validation accuracy: {100 * correct / total}%')
            logger(logger_name, f'Validation accuracy: {100 * correct / total}%')

        # Save the model after each epoch
        # if (epoch + 1) % 1 == 0:
        #     folder_path=args.dataset+'_'+args.model_type+'_'+args.mae_type
        #     path = f'./{folder_path}/'

        #     # Make the directory if it doesn't exist
        #     os.makedirs(path, exist_ok=True)

        #     # Save the model
        #     torch.save(model.state_dict(), os.path.join(path, f'_model_epoch_{epoch+1}.pth'))



if __name__ == '__main__':
    args = args_parser()
    print(args)
    train_model(args)


# nohup python -u trainer.py --dataset 'cifar100' --model_type 'sleepnet' > cifar100_sleepnet_google.txt 2>&1 &
# 3780186
# nohup python -u trainer.py --dataset 'cifar100' --model_type 'resnet' > cifar100_resnet50_google.txt 2>&1 &
# 3146119
# nohup python -u trainer.py --dataset 'cifar100' --model_type 'vit' > cifar100_vit_google.txt 2>&1 &
# 3314000
# nohup python -u trainer.py --dataset 'imagenet' --model_type 'sleepnet' > imagenet_sleepnet_google.txt 2>&1 &
# nohup python -u trainer.py --dataset 'imagenet' --model_type 'resnet' > imagenet_resnet50_google.txt 2>&1 &
# nohup python -u trainer.py > dreamnet.txt 2>&1 &
