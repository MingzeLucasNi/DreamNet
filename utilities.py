import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
class CustomImageNet(Dataset):
    def __init__(self, split='train'):
        super().__init__()

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the same size as inputs to VIT or ResNet (224x224)
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters from ImageNet
        ])

        # Load the dataset
        self.dataset = load_dataset("imagenet-1k", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = Image.open(sample['file']).convert('RGB')  # Assuming the images are RGB
        image = self.transform(image)
        label = torch.tensor(sample['label'])

        return image, label




def get_dataset(name, train=True, transform=None):
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif name == 'imagenet':
        return CustomImageNet(split='train' if train else 'val')
    else:
        raise ValueError(f'Invalid dataset name: {name}')
    
def get_mae(name):
    if name=='google':
        mae="google/vit-large-patch16-224"
    elif name=='openai':
        mae="openai/clip-vit-large-patch14"
    else:
        mae="facebook/vit-mae-base"
    return mae

import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
class CustomImageNet(Dataset):
    def __init__(self, split='train'):
        super().__init__()

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the same size as inputs to VIT or ResNet (224x224)
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters from ImageNet
        ])

        # Load the dataset
        self.dataset = load_dataset("imagenet-1k", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = Image.open(sample['file']).convert('RGB')  # Assuming the images are RGB
        image = self.transform(image)
        label = torch.tensor(sample['label'])

        return image, label




def get_dataset(name, train=True, transform=None):
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif name == 'imagenet':
        return CustomImageNet(split='train' if train else 'val')
    else:
        raise ValueError(f'Invalid dataset name: {name}')
    
def get_mae(name):
    if name=='google':
        mae="google/vit-large-patch16-224"
    elif name=='openai':
        mae="openai/clip-vit-large-patch14"
    else:
        mae="facebook/vit-mae-base"
    return mae