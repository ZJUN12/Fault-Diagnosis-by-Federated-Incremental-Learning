import torch
import math
import itertools
import torch.nn as nn
from args.args import args_parser
import torch.nn.functional as F


args = args_parser()

class network(nn.Module):
    
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
    
    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias
        
    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)
    
    
def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


#-------------------------Lightweight Model: MobileNet-------------------------
class DepthwiseSeparableConv1D(nn.Module):
    expansion = 1  
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=args.numclass):
        super(MobileNet, self).__init__()
        self.linear = nn.Linear(36, 16 * 36)

        self.conv1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.in_channels = 32
        self.layer1 = self._make_layer(DepthwiseSeparableConv1D, 64, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(DepthwiseSeparableConv1D, 128, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(DepthwiseSeparableConv1D, 128, num_blocks=1, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.linear(x)
        x = x.view(x.size(0), 16, -1)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x