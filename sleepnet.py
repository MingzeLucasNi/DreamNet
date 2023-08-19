import torch.nn as nn
from transformers import ViTModel
import torch
mae = ViTModel.from_pretrained("google/vit-base-patch16-224",output_hidden_states=True)

# Freezing the parameters
for param in mae.parameters():
    param.requires_grad = False

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, MAE=mae):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.mae = MAE
        self.linear = nn.Linear(768, 64*56*56)
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        # self.conv1x1_resize = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv1x1_resize = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x
        x_reshaped = self.upsample(x)
        x_reshaped = self.conv1x1(x_reshaped)
        hidden = self.mae(x_reshaped).last_hidden_state
        hidden = hidden.mean(dim=1)
        # print('the size after mean:',hidden.shape)
        hidden = self.linear(hidden)
        hidden = hidden.view(x.size(0), 64,56,56)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        # print('out size:',out.shape)
        # hidden_reshaped = hidden.view(out.size())
        out = out+hidden
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, MAE=mae):
        super(ResNet, self).__init__()
        # self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=75)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 , num_classes)
        self.mae = MAE


    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # hidden = self.mae(x).hidden_states
        # logits = self.mae(x).logits
        # logits = self.mae.unpatchify(logits)
        # hidden = hidden[-1]
        # hidden = hidden.view(hidden.size(0), -1)
        # hidden = self.maxpool2(hidden)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def get_dataset(name, train=True, transform=None):
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif name == 'imagenet':
        return torchvision.datasets.ImageNet(root='./data', split='train' if train else 'val', download=True, transform=transform)
    else:
        raise ValueError(f'Invalid dataset name: {name}')



##text
import torch.nn as nn
from transformers import BertModel

class TextCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_size, num_filters, filter_sizes, dropout=0.5, pretrain_model='bert-base-uncased'):
        super(TextCNN, self).__init__()
        
        # Load BERT model for token embedding
        self.bert = BertModel.from_pretrained(pretrain_model)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size))
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.fc2 = nn.Linear(512*768, 96)
        
    def forward(self, input_ids):
        sleeps = self.bert(input_ids=input_ids).last_hidden_state
        x = sleeps.unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        print('x size:',x.shape)
        print('sleeps size:',sleeps.shape)
        hidden=sleeps.view(sleeps.size(0),-1)
        hides=self.fc2(hidden)
        x=x+hides
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
