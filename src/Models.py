# -*- coding: utf-8 -*-
# ==============================================================================
# Multi-Model Classifier (多模型分类器) - Models Definitions
# 此文件包含所有支持的模型结构，支持动态输入通道(IN_CHANNELS)和类别数(NUM_CLASSES)
# This file contains all supported model architectures with dynamic channels and classes.
# 方便初学者学习和 DIY 修改！(Easy for beginners to learn and DIY!)
# ==============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------------------------------------------------------
# 1. C1_LeNet5 (经典 LeNet-5 网络)
# ------------------------------------------------------------------------------
class C1_LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, img_size=128):
        super(C1_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 动态计算全连接层的输入维度
        self.fc_input_dim = self._get_conv_output(in_channels, img_size)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def _get_conv_output(self, in_channels, img_size):
        x = torch.zeros(1, in_channels, img_size, img_size)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------------------------------------------------------
# 2. C2_AlexNet (AlexNet)
# ------------------------------------------------------------------------------
class C2_AlexNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C2_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 3. C3_VGG16 (VGG-16)
# ------------------------------------------------------------------------------
class C3_VGG16(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C3_VGG16, self).__init__()
        vgg16 = models.vgg16(weights=None)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            *list(vgg16.features.children())[1:]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 4. C4_VGG19 (VGG-19)
# ------------------------------------------------------------------------------
class C4_VGG19(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C4_VGG19, self).__init__()
        vgg19 = models.vgg19(weights=None)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            *list(vgg19.features.children())[1:]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 5. C5_ResNet18 (ResNet-18)
# ------------------------------------------------------------------------------
class C5_ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C5_ResNet18, self).__init__()
        resnet18 = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 6. C6_ResNet50 (ResNet-50)
# ------------------------------------------------------------------------------
class C6_ResNet50(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C6_ResNet50, self).__init__()
        resnet50 = models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 7. C7_DenseNet121 (DenseNet-121)
# ------------------------------------------------------------------------------
class C7_DenseNet121(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C7_DenseNet121, self).__init__()
        densenet = models.densenet121(weights=None)
        self.features = densenet.features
        self.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ------------------------------------------------------------------------------
# 8. C8_MobileNetV2 (MobileNet-V2)
# ------------------------------------------------------------------------------
class C8_MobileNetV2(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C8_MobileNetV2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(weights=None)
        self.mobilenetv2.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mobilenetv2(x)

# ------------------------------------------------------------------------------
# 9. C9_SimpleCNN (自定义简单卷积网络)
# ------------------------------------------------------------------------------
class C9_SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, img_size=128):
        super(C9_SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------------------------------------------------------------
# 10. C10_MlpMixer (简易 MLP-Mixer)
# ------------------------------------------------------------------------------
class C10_MlpMixer(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, img_size=128):
        super(C10_MlpMixer, self).__init__()
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.dim = 256
        
        self.patch_embed = nn.Conv2d(in_channels, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 简化版只用了一层全连接代替复杂的混合层
        self.mixer_layer = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        self.fc = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, dim)
        x = x + self.mixer_layer(x)
        x = x.mean(dim=1) # (B, dim)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 11. C11_GhostNet (简易版 GhostNet 模块组装)
# ------------------------------------------------------------------------------
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GhostModule, self).__init__()
        half_out = out_channels // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, half_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_out),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(half_out, half_out, kernel_size=3, padding=1, groups=half_out, bias=False),
            nn.BatchNorm2d(half_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class C11_GhostNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C11_GhostNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.ghost1 = GhostModule(16, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.ghost2 = GhostModule(32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.ghost1(x))
        x = self.ghost2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 12. C12_DepthwiseCNN (深度可分离卷积网络)
# ------------------------------------------------------------------------------
class C12_DepthwiseCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C12_DepthwiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        
        # Depthwise Separable
        self.depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 13. C13_ConvMixer (简易版 ConvMixer)
# ------------------------------------------------------------------------------
class C13_ConvMixer(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, dim=256, depth=4):
        super(C13_ConvMixer, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=5, groups=dim, padding=2),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# Helper Function to Get Models (获取模型字典的辅助函数)
# ------------------------------------------------------------------------------
def get_all_models(in_channels, num_classes, img_size):
    """
    Returns a dictionary of all initialized models.
    返回所有初始化后的模型字典。
    """
    return {
        "C1_LeNet5": C1_LeNet5(in_channels, num_classes, img_size),
        "C2_AlexNet": C2_AlexNet(in_channels, num_classes),
        "C3_VGG16": C3_VGG16(in_channels, num_classes),
        "C4_VGG19": C4_VGG19(in_channels, num_classes),
        "C5_ResNet18": C5_ResNet18(in_channels, num_classes),
        "C6_ResNet50": C6_ResNet50(in_channels, num_classes),
        "C7_DenseNet121": C7_DenseNet121(in_channels, num_classes),
        "C8_MobileNetV2": C8_MobileNetV2(in_channels, num_classes),
        "C9_SimpleCNN": C9_SimpleCNN(in_channels, num_classes, img_size),
        "C10_MlpMixer": C10_MlpMixer(in_channels, num_classes, img_size),
        "C11_GhostNet": C11_GhostNet(in_channels, num_classes),
        "C12_DepthwiseCNN": C12_DepthwiseCNN(in_channels, num_classes),
        "C13_ConvMixer": C13_ConvMixer(in_channels, num_classes)
    }
