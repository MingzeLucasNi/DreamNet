import torch.nn as nn
from transformers import ViTMAEForPreTraining
import torch
mae =  ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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
        # self.linear = nn.Linear(768, 128*56*56)
        # self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        # x_reshaped = self.upsample(x)
        # x_reshaped = self.conv1x1(x_reshaped)
        # hidden = mae(x_reshaped).logits
        # reconstruction=mae.unpatchify(hidden)
        # hidden = hidden.mean(dim=1)
        # hidden = self.linear(hidden)
        # hidden = hidden.view(x.size(0), 128,56,56)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.downsample(identity)
        # out = out+hidden
        out = self.relu(out)
        return out

class SleepConnection(nn.Module):
    def __init__(self, out_channels=128, target_shape=(56, 56), mae=mae):
        super(SleepConnection, self).__init__()
        # self.upsample = nn.Upsample(target_shape, mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(3, out_channels, kernel_size=1, stride=1, padding=0)
        self.mae = mae
        for param in self.mae.parameters():
            param.requires_grad = False
        # self.linear = nn.Linear(768, out_channels * target_shape[0] * target_shape[1])
        self.linear = nn.Linear(768, 128*56*56)
        self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x_reshaped = self.upsample(x)
        x_reshaped = self.conv1x1(x_reshaped)
        hidden = self.mae(x_reshaped).logits
        reconstruction=self.mae.unpatchify(hidden)
        hidden = hidden.mean(dim=1)
        hidden = self.linear(hidden)
        hidden = hidden.view(x.size(0), 128,56,56)
        return hidden, reconstruction  # Return both hidden states and mae logits as needed




class DreamNet(nn.Module):
    def __init__(self, block, connection,num_blocks, num_classes=1000):
        super(DreamNet, self).__init__()
        # self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.in_channels = 128
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=75)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.compressor = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.connect=connection(mae=mae)
        self.linear = nn.Linear(768, 128*56*56)
        self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)

        # self.layer1 = block(128, 128, stride=1)
        # self.layer2 = block(128, 128, stride=1)
        # self.layer3 = block(128, 128, stride=1)
        # self.layer4 = block(128, 128, stride=1)

        # self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 , num_classes)
        # self.mae = MAE


    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    

    

    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # Initialize a list to hold reconstruction outputs
        reconstructions = []

        # Process layers with sleep connection
        for layer in [self.layer1, self.layer2, self.layer3]:
            out = layer(out)
            hidden, reconstruction = self.connect(out)
            reconstructions.append(reconstruction)
            out = out + hidden  # Ensure this is not considered as an inplace operation


        # mean all reconstructions for the "dream" phase
        reconstruction = torch.stack(reconstructions).mean(dim=0)


        # Start to dream
        dream = reconstruction
        dream = self.relu(self.bn2(self.conv2(dream)))
        dream = self.maxpool(dream)
        dream= self.compressor(dream)
        # for layer in [self.layer1, self.layer2]:
        #     dream = layer(dream)
        #     hidden, _ = self.connect(dream)
        #     # reconstructions.append(reconstruction)
        #     dream = dream + hidden  # Ensure this is not considered as an inplace operation
        # Finish dream
        # new_out = dream
        new_out = dream + out  # Make sure this is not inplace
        out = self.avgpool(new_out)  # Use new_out instead of out to avoid potential inplace modification
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



def ResNet18(num_classes):
    return DreamNet(BasicBlock, SleepConnection, [4, 4, 2, 2], num_classes)
