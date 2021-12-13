import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch.nn as nn 
import torch.nn.functional as F
import torch


class BottleNeck(nn.Module):
    expansion = 4 #블록의 마지막 레이어에서 출력 차원을 4배로 늘린다
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #Resnet50은 1x1, 3x3, 1x1을 거치는 3개의 컨볼루션 레이어가 하나의 블록으로 구성된다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential() #x를 그대로 사용

        self.relu = nn.ReLU()

        #stride가 2 이상이어서 feature map의 크기가 다르거나, 출력층의 채널수가 다른 경우 1x1컨볼루션을 이용해 x의 크기를 맞춰준다.
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=2, init_weights=True):
        super().__init__()

        self.in_channels=64

        #7x7의 컨볼루션 레이어와 맥스풀링
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        #3개의 블록이 있으며 3x3 컨볼루션 레이어에서도 stride를 1로 두어 output size에 변화가 없다.
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        #conv3~5에서는 각각 4,6,3개의 블록이 있으며 3x3 컨볼루션 레이어에서 stride를 2로 두어 output size가 절반으로 줄어든다.
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        #Average Pooling을 통해 각 채널에서의 평균값 추출
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        #마지막 출력층의 채널 수인 512*4개가 input으로 들어가 클래수의 수만큼 출력한다.
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #가중치 초기화
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])