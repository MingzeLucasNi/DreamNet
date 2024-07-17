import torch.nn as nn
from transformers import ViTMAEForPreTraining
import torch
mae =  ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae.to(device)
# Freezing the parameters
# for param in mae.parameters():
#     param.requires_grad = False

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
        # self.mae = MAE
        self.linear = nn.Linear(768, 64*56*56)
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        # self.conv1x1_resize = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv1x1_resize = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x
        # print('the x size:',x.shape)
        x_reshaped = self.upsample(x)
        # print('the x reshaped size:',x_reshaped.shape)
        x_reshaped = self.conv1x1(x_reshaped)
        # print('the x reshaped size:',x_reshaped.shape)
        hidden = mae(x_reshaped).logits
        # print('the size ater mae:',hidden.shape)
        reconstruction=mae.unpatchify(hidden)
        hidden = hidden.mean(dim=1)
        # print('the size after mae:',hidden.shape)
        # print('the size after mean:',hidden.shape)
        hidden = self.linear(hidden)
        # print('the size after mae:',hidden.shape)
        hidden = hidden.view(x.size(0), 64,56,56)
        # print('the size after mae:',hidden.shape)
        out = self.relu(self.bn1(self.conv1(x)))
        # print('out size:',out.shape)
        out = self.bn2(self.conv2(out))
        # print('out size:',out.shape)
        out += self.downsample(identity)
        # print('out size:',out.shape)
        # hidden_reshaped = hidden.view(out.size())
        # print('hidden size:',hidden.shape)

        out = out+hidden
        out = self.relu(out)
        return out, reconstruction



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        # self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=75)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)

        self.layer1 = block(64, 64, stride=1)
        self.layer2 = block(64, 64, stride=1)
        self.layer3 = block(64, 64, stride=1)
        # self.layer4 = block(64, 64, stride=1)

        # self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 , num_classes)
        # self.mae = MAE


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
        # print('the size after maxpool:',out.shape)
        out, d1 = self.layer1(out)
        # print('the size after layer1:',out.shape)
        out, d2 = self.layer2(out)
        out, d3 = self.layer3(out)
        # out, d4 = self.layer4(out)


        new_out = x+d1+d2+d3
        out = self.relu(self.bn1(self.conv1(new_out)))
        out = self.maxpool(out)
        out, d1 = self.layer1(out)
        out, d2 = self.layer2(out)
        out, d3 = self.layer3(out)
        # out, d4 = self.layer4(out)


        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# 420505168