# models/simsiam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class BatchNorm1d(nn.Module):
    """批归一化层实现,与论文代码保持一致"""

    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class projection_MLP(nn.Module):
    """投影头MLP实现"""

    def __init__(self, in_dim, out_dim=2048, num_layers=3):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 注意最后一层的BatchNorm不带affine参数
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    """预测头MLP实现"""

    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim  # 输出维度与输入相同
        hidden_dim = int(out_dim / 4)  # 隐层维度是输出维度的1/4

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    """SimSiam模型实现"""

    def __init__(self, emam, args):
        super(SimSiam, self).__init__()
        self.emam = emam  # 动量系数

        # 创建backbone及其动量编码器副本
        self.backbone = self.get_backbone(args.arch)
        self.backbone_k = self.get_backbone(args.arch)

        # 获取特征维度
        dim_out, dim_in = self.backbone.fc.weight.shape
        dim_mlp = 2048

        # 移除原始fc层
        self.backbone.fc = nn.Identity()
        self.backbone_k.fc = nn.Identity()

        # 创建投影头及其动量编码器副本
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_mlp),
            BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )
        self.projector_k = nn.Sequential(
            nn.Linear(dim_in, dim_mlp),
            BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(dim_out, 512),
            BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, dim_out)
        )

        # 整体编码器
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )

        # 分类器
        self.linear = nn.Linear(dim_in, args.num_classes)
        self.classifier = nn.Sequential(
            self.backbone,
            self.linear
        )

        # 初始化动量编码器
        for param_q, param_k in zip(self.encoder.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @staticmethod
    def get_backbone(backbone_name):
        """获取backbone网络"""
        return {
            'resnet18': ResNet18(),
            'resnet34': ResNet34(),
            'resnet50': ResNet50(),
            'resnet101': ResNet101(),
            'resnet152': ResNet152()
        }[backbone_name]

    def forward(self, im_aug1, im_aug2, img_weak):
        """前向传播
        Args:
            im_aug1: 第一个强增强视图
            im_aug2: 第二个强增强视图
            img_weak: 弱增强视图
        """
        # 分类输出
        output = self.classifier(img_weak)

        # 对比学习分支
        z1 = self.encoder(im_aug1)
        p1 = self.predictor(z1)
        p1 = F.normalize(p1, dim=1)

        # 更新动量编码器
        with torch.no_grad():
            m = self.emam
            for param_q, param_k in zip(self.encoder.parameters(),
                                        self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

        # 获取另一个视图的表征
        z2 = self.encoder_k(im_aug2)
        z2 = F.normalize(z2, dim=1)

        return p1, z2, output

    def forward_test(self, x):
        """测试时只使用分类器"""
        return self.classifier(x)