"""
Module creates a model using the ResNet convolutional neural network architecture
"""

import torch
import torch.nn as nn

__all__ = {
    "ResBlock",
    "ResNet"
}

# Normalizes data
def norm_data(proj):
    proj = (proj - torch.min(proj)) / (torch.max(proj) - torch.min(proj))
    return proj

# Creates residual blocks for ResNet
class ResBlock(nn.Module):

    # Indicates number of feature maps through convolution layer
    expansion = 1

    # Initializing residual block
    def __init__(self, in_planes, out_planes, stride = 1, downsample = None, groups = 1, 
                base_width = 64, dilation = 1, norm_layer = None):
        super(ResBlock, self).__init__

        # Sets default normalization layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Structure of residual block
        self.conv1 = nn.Conv2d(in_planes, out_planes, stride, kernel_size = 3,
                               stride = stride, padding = dilation, groups = groups,
                               bias = False, dilation = dilation)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, stride, kernel_size = 3,
                               stride = stride, padding = dilation, groups = groups,
                               bias = False, dilation = dilation)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.stride = stride

    # Forward propogation of block
    def forward(self, x):
        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity = self.downsample(x)

        # F(x) + x
        output += identity
        output = self.relu(output)

        return output
    

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 2, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None):
        super(ResNet, self).__init__()

        # Sets default normalization layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Replaces stride with dilated convolution
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False]

        # Error if length of dilation is not equal to amount of dilations needed
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.in_planes = 64
        self.base_width = width_per_group
        self.groups = groups

        # Initial 7x7 convolution layer
        self.conv1 = nn.Conv2d(2, self.in_planes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # Three residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, dilate = replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2, dilate = replace_stride_with_dilation[1])

        # Applying adaptive average pool and create fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
    
    # Creates layer from residual block
    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size = 1, stride = stride),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups,
                            base_width = self.base_width, norm_layer = norm_layer))
        
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride, downsample, self.groups,
                            base_width = self.base_width, norm_layer = norm_layer))
            
        return nn.Sequential(*layers)
    
    # Forward propogation of resnet
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim = 1) # concatenates moving and fixed projections
        x = norm_data(x)

        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Average pool and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
# creates resnet model
def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model
    
# calls resnet18
def resnet18(**kwargs):
    return _resnet('resnet18', ResBlock, [2, 2, 2, 2], **kwargs)