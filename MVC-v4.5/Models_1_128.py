import os
current_dir = os.getcwd()
import time
start_time = time.time()
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
#         if self.reduction == 'mean':
#             return focal_loss
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss, ce_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            # 根据目标标签的类别数动态调整 alpha 参数的维度
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets]
        else:
            alpha = 1

        focal_loss = (alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        if self.reduction == 'mean':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss, ce_loss


class OhemCELoss(nn.Module):
    def __init__(self, threshold, min_kept, ignore_label):
        super(OhemCELoss, self).__init__()
        self.threshold = threshold
        self.min_kept = min_kept
        self.ignore_label = ignore_label
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def forward(self, logits, targets):
        # 计算交叉熵损失
        ce = self.ce_loss(logits, targets)

        # 排除忽略标签对损失的影响
        valid_mask = targets != self.ignore_label
        ce = ce[valid_mask]

        # 根据置信度进行难例挖掘
        sorted_ce, _ = torch.sort(ce, descending=True)
        num_kept = min(max(self.min_kept, sorted_ce.numel()), sorted_ce.numel())
        threshold = sorted_ce[num_kept - 1] if num_kept > 0 else 0.0
        hard_mask = ce >= threshold

        # 仅保留难例样本的损失
        ohem_ce = ce[hard_mask]
        return ohem_ce.mean()

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)
        return ce_loss(input, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, num_classes, beta=0.999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.class_weights = None

    def compute_weights(self, labels):
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        total_samples = class_counts.sum()
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        return weights

    def forward(self, inputs, targets):
        weights = self.compute_weights(targets)
        loss = F.cross_entropy(inputs, targets, weight=weights)
        return loss

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, num_classes, beta=0.8):
        super(BalancedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.weights = nn.Parameter(torch.ones(num_classes))

    def forward(self, inputs, targets):
        # 计算类别的频率
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        class_freq = class_counts / torch.sum(class_counts)

        # 计算类别权重
        weights = 1.0 - torch.pow(self.beta, class_freq)
        self.weights.data = weights

        # 计算 softmax
        probs = F.softmax(inputs, dim=1)

        # 计算交叉熵损失
        loss = F.nll_loss(torch.log(probs), targets)
        loss = torch.mean(loss * self.weights)
        return loss


class EqualizationLoss(nn.Module):
    def __init__(self, num_classes):
        super(EqualizationLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        # 计算每个类别的样本数量
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()

        # 计算每个类别的权重
        weights = outputs.size(0) / (class_counts * self.num_classes)

        # 计算交叉熵损失
        loss = F.cross_entropy(outputs, targets, reduction='none')

        # 对损失进行加权平均
        equalization_loss = torch.sum(weights[targets] * loss) / outputs.size(0)

        return equalization_loss


class LDAMLoss(nn.Module):
    def __init__(self, num_classes, max_m=0.5, weight=None):
        super(LDAMLoss, self).__init__()
        assert 0 < max_m <= 1, f"max_m must be in range (0, 1], got {max_m}"
        self.num_classes = num_classes
        self.max_m = max_m
        self.weight = weight
    def forward(self, logits, targets):
        assert logits.size(0) == targets.size(0), "logits and targets must have the same size"
        # 计算每个类别的样本数量
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        # 计算每个类别的频率
        class_freqs = class_counts / torch.sum(class_counts)
        # 计算每个类别的权重
        weights = (1 - class_freqs) ** self.max_m
        weights /= torch.sum(weights)
        # 使用交叉熵损失
        loss = F.cross_entropy(logits, targets, weight=weights, reduction='mean')
        return loss




# 定义分类模型C1_LeNet5 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C1_LeNet5(nn.Module):
    def __init__(self, num_classes=4):
        super(C1_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)  # 修改输入维度为16 * 29 * 29
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)  # 调整为正确的输入维度
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义分类模型C2_AlexNet ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C2_AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(C2_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
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




# 定义分类模型C3_VGG16 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C3_VGG16(nn.Module):
    def __init__(self, num_classes=4):
        super(C3_VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            *list(vgg16.features.children())[1:]
        )
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
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
        x = torch.flatten(x, 1)  # 将特征图展平为向量
        x = self.classifier(x)
        return x

# 定义分类模型C3_VGG19 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C3_VGG19(nn.Module):
    def __init__(self, num_classes=4):
        super(C3_VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            *list(vgg19.features.children())[1:]
        )
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 定义分类模型C4_GoogLeNetV1 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1proj):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce3x3, out3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce5x5, out5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        return torch.cat(outputs, 1)


class C4_GoogLeNetV1(nn.Module):
    def __init__(self, num_classes=4):
        super(C4_GoogLeNetV1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception3 = nn.Sequential(
            InceptionModule(192, 64, 96, 128, 16, 32, 32),
            InceptionModule(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception4 = nn.Sequential(
            InceptionModule(480, 192, 96, 208, 16, 48, 64),
            InceptionModule(512, 160, 112, 224, 24, 64, 64),
            InceptionModule(512, 128, 128, 256, 24, 64, 64),
            InceptionModule(512, 112, 144, 288, 32, 64, 64),
            InceptionModule(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception5 = nn.Sequential(
            InceptionModule(832, 256, 160, 320, 32, 128, 128),
            InceptionModule(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 定义分类模型C5_ResNet18 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C5_ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(C5_ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


# 定义分类模型C5_ResNet50 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C5_ResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(C5_ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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



# 定义分类模型C5_ResNet101 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C5_ResNet101(nn.Module):
    def __init__(self, num_classes=4):
        super(C5_ResNet101, self).__init__()
        resnet101 = models.resnet101(pretrained=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet101.bn1
        self.relu = resnet101.relu
        self.maxpool = resnet101.maxpool
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4
        self.avgpool = resnet101.avgpool
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


# 定义分类模型C6_ResNext ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C6_ResNeXt(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(C6_ResNeXt, self).__init__()

        resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnext50_32x4d.bn1
        self.relu = resnext50_32x4d.relu
        self.maxpool = resnext50_32x4d.maxpool

        self.layer1 = resnext50_32x4d.layer1
        self.layer2 = resnext50_32x4d.layer2
        self.layer3 = resnext50_32x4d.layer3
        self.layer4 = resnext50_32x4d.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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



# 定义分类模型C7_DenseNet121 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C7_DenseNet121(nn.Module):
    def __init__(self, num_classes=4):
        super(C7_DenseNet121, self).__init__()

        densenet = models.densenet121(pretrained=False)
        self.features = densenet.features

        # Modify the first convolution layer to accept single channel input
        self.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 定义分类模型C7_DenseNet161 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C7_DenseNet161(nn.Module):
    def __init__(self, num_classes=4):
        super(C7_DenseNet161, self).__init__()

        densenet = models.densenet161(pretrained=False)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2208, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 定义分类模型C7_DenseNet ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C7_DenseNet201(nn.Module):
    def __init__(self, num_classes=4):
        super(C7_DenseNet201, self).__init__()

        densenet = models.densenet201(pretrained=False)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1920, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



# 定义分类模型C8_MobileNetV2 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C8_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(C8_MobileNetV2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=False)
        # 修改最后的全连接层，使其输出为你的类别数
        self.mobilenetv2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenetv2.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenetv2(x)

# 定义分类模型C8_MobileNetV3 ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class C8_MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(C8_MobileNetV3, self).__init__()
        self.mobilenetv3 = models.mobilenet_v3_small(pretrained=False)
        # 修改第一个卷积层的输入通道数
        self.mobilenetv3.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # 修改最后的全连接层，使其输出为类别数
        self.mobilenetv3.classifier[3] = nn.Linear(self.mobilenetv3.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.mobilenetv3(x)


# 定义分类模型C9_EfficientNet ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio=16):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1
        expanded_channels = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(Swish())

        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
            SEBlock(expanded_channels, reduction_ratio=reduction_ratio),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class C9_EfficientNet(nn.Module):
    def __init__(self, num_classes=4, width_multiplier=1.0, depth_multiplier=1.0):
        super(C9_EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier

        # Define settings for different EfficientNet variants
        settings = [
            # t, c, n, s, k, r
            [1, 16, 1, 1, 3, 1],
            [6, 24, 2, 2, 3, 2],
            [6, 40, 2, 2, 5, 2],
            [6, 80, 3, 2, 3, 2],
            [6, 112, 3, 1, 5, 1],
            [6, 192, 4, 2, 5, 2],
            [6, 320, 1, 1, 3, 1]
        ]

        # Calculate number of channels based on width multiplier
        channels = [int(round(ch * width_multiplier)) for ch in [32, 16, 24, 40, 80, 112, 192, 320]]

        # Calculate number of layers based on depth multiplier
        num_layers = [int(round(n * depth_multiplier)) for n in [1, 2, 2, 3, 3, 4, 4, 1]]

        # Build the network
        layers = []
        layers.append(nn.Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels[0]))
        layers.append(Swish())

        in_channels = channels[0]
        for i in range(len(settings)):
            t, c, n, s, k, r = settings[i]
            out_channels = channels[i + 1]
            layers.append(self._make_layer(in_channels, out_channels, t, c, n, s, k, r))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels, expand_ratio, channels, num_blocks, stride, kernel_size, reduction_ratio):
        layers = []
        layers.append(MBConvBlock(in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio))
        for _ in range(num_blocks - 1):
            layers.append(MBConvBlock(out_channels, out_channels, expand_ratio, kernel_size, 1, reduction_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# 定义分类模型C10_DPN ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class DualPathBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DualPathBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class C10_DPN(nn.Module):
    def __init__(self, num_classes=4):
        super(C10_DPN, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(DualPathBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(DualPathBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(DualPathBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, channels, stride=stride))
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# 定义分类模型C11_IGCNet ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class IGCNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(IGCNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return torch.cat([out1, out2, out3], dim=1)

class C11_IGCNet(nn.Module):
    def __init__(self, num_classes=4):
        super(C11_IGCNet, self).__init__()
        self.conv1 = IGCNetBlock(3, 64)
        self.conv2 = IGCNetBlock(192, 128)
        self.conv3 = IGCNetBlock(384, 256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 修改这里，使输入维度与实际输出维度相匹配
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape)  # 打印中间输出的形状
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(x.shape)  # 打印展平后的形状
        x = self.fc(x)
        return x

# 定义分类模型C12_ ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class DarknetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DarknetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class C12_Darknet53(nn.Module):
    def __init__(self, num_classes=4):
        super(C12_Darknet53, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layers(in_channels=32, out_channels=64, blocks=1)
        self.layer2 = self.make_layers(in_channels=64, out_channels=128, blocks=2)
        self.layer3 = self.make_layers(in_channels=128, out_channels=256, blocks=8)
        self.layer4 = self.make_layers(in_channels=256, out_channels=512, blocks=8)
        self.layer5 = self.make_layers(in_channels=512, out_channels=1024, blocks=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def make_layers(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(DarknetBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(DarknetBlock(out_channels, out_channels // 2))
            in_channels = out_channels
            out_channels //= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x