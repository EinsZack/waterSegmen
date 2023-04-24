import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet101_Weights


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3Plus, self).__init__()

        # Atrous Convolutional backbone
        self.resnet_features = ResNet101()

        # ASPP module
        self.aspp = ASPPModule()

        # Upsampling module
        self.upsample = UpsampleModule()

        # DeepLab head
        self.classifier = ClassifierModule(n_classes)

    def forward(self, x):
        size = x.size()[2:]

        # Atrous Convolutional backbone
        x, low_level_features = self.resnet_features(x)

        # ASPP module
        x = self.aspp(x)

        # Upsampling module
        x = self.upsample(x, low_level_features)

        # DeepLab head
        x = self.classifier(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()

        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
        )
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x):
        size = x.size()[2:]

        x = self.layer1(x)
        low_level_features = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


class ASPPModule(nn.Module):
    def __init__(self):
        super(ASPPModule, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(2048, 48, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(48)
        self.conv_1x1_4 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.size()[2:]

        # Image-level features
        image_features = self.avg_pool(x)
        image_features = self.conv_1x1_2(image_features)
        image_features = self.bn_conv_1x1_2(image_features)
        image_features = F.interpolate(image_features, size, mode='bilinear', align_corners=True)

        # Atrous Convolutional features
        atrous_features = self.conv_1x1_1(x)
        atrous_features = self.bn_conv_1x1_1(atrous_features)

        atrous_features_1 = self.conv_3x3_1(x)
        atrous_features_1 = self.bn_conv_3x3_1(atrous_features_1)

        atrous_features_2 = self.conv_3x3_2(x)
        atrous_features_2 = self.bn_conv_3x3_2(atrous_features_2)

        atrous_features_3 = self.conv_3x3_3(x)
        atrous_features_3 = self.bn_conv_3x3_3(atrous_features_3)

        # Concatenate features
        features = torch.cat([atrous_features, atrous_features_1, atrous_features_2, atrous_features_3, image_features],
                             dim=1)
        features = self.conv_1x1_4(features)
        features = self.bn_conv_1x1_4(features)
        features = self.dropout(features)

        return features


class UpsampleModule(nn.Module):
    def __init__(self):
        super(UpsampleModule, self).__init__()
        self.conv = nn.Conv2d(256, 48, kernel_size=1)
        self.bn = nn.BatchNorm2d(48)

    def forward(self, x, low_level_features):
        size = low_level_features.size()[2:]

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        low_level_features = self.conv(low_level_features)
        low_level_features = self.bn(low_level_features)
        x = torch.cat([x, low_level_features], dim=1)

        return x


class ClassifierModule(nn.Module):
    def __init__(self, n_classes):
        super(ClassifierModule, self).__init__()

        self.conv = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
