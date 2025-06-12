import torch
import torch.nn as nn
from torchvision import models


class ResNetModel(nn.Module):
    def __init__(self, arch='resnet18', num_classes=2880):
        """初始化ResNet模型

        Args:
            arch: 架构类型 'resnet18'/'resnet34'/'resnet50'
            num_classes: 类别数量
        """
        super(ResNetModel, self).__init__()

        # 根据架构类型创建模型
        if arch == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif arch == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.feature_dim = 512
        elif arch == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError(f'不支持的ResNet架构: {arch}')

        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 添加新的分类层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, mode='train'):
        """前向传播

        Args:
            x: 输入张量
            mode: 'train' 或 'test'
        """
        # 提取特征
        features = self.features(x)
        features = features.view(features.size(0), -1)  # 展平特征

        # 分类
        out = self.fc(features)
        return out

    def get_features(self, x):
        """提取特征向量"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            features = self.features(x)
            features = features.view(features.size(0), -1)  # [B, feature_dim]
            return features


def create_model(args, num_classes=None):
    """创建模型工厂函数

    Args:
        args: 配置参数
        num_classes: 类别数(可选)
    Returns:
        model: ResNet模型
    """
    if num_classes is None:
        num_classes = args.num_classes // args.num_clients

    model = ResNetModel(
        arch=args.arch,
        num_classes=num_classes
    )
    return model.to(args.device)