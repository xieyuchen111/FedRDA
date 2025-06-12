# utils/simclr_pretraining.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, List, Tuple
import copy
import numpy as np
import time

class ContrastiveTransform:
    """为指静脉数据集定制的SimCLR数据增强"""
    def __init__(self):
        self.augment = transforms.Compose([
            # 移除这里的ToTensor和Normalize,因为输入已经是tensor
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.2, 1.0),
                ratio=(3./4., 4./3.),
                 antialias=True  # 添加此参数
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23)
        ])

    def __call__(self, x):
        return self.augment(x), self.augment(x)


class ContrastiveDataset(Dataset):
    """对比学习数据集包装器"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transform = ContrastiveTransform()

    def __getitem__(self, idx):
        img, label, idx, orig_label = self.base_dataset[idx]
        if not isinstance(img, torch.Tensor):
            # 如果输入不是tensor,先转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            img = transform(img)
        x_i, x_j = self.transform(img)
        return (x_i, x_j), label, idx, orig_label

    def __len__(self):
        return len(self.base_dataset)


class ProjectionHead(nn.Module):
    """投影头 - 将512维特征映射到128维对比学习空间"""

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.projection(x)
        return F.normalize(x, dim=1)


class SimCLR(nn.Module):
    """SimCLR模型"""

    def __init__(self, backbone, feature_dim=512, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(feature_dim, projection_dim)

        # 冻结backbone的BN层
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def forward(self, x1, x2):
        # 获取backbone特征
        h1 = self.backbone.features(x1).squeeze()  # [N, feature_dim]
        h2 = self.backbone.features(x2).squeeze()  # [N, feature_dim]

        # 投影
        z1 = self.projection(h1)  # [N, projection_dim]
        z2 = self.projection(h2)  # [N, projection_dim]

        return h1, h2, z1, z2

    def get_features(self, x):
        try:
            return self.backbone.features(x).squeeze()
        except Exception as e:
            logging.error(f"特征提取时出错: {str(e)}")
            raise e

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        try:
            super().__init__()
            self.temperature = temperature
        except Exception as e:
            logging.error(f"初始化 InfoNCELoss 时出错: {str(e)}")
            raise e

    def forward(self, z1, z2):
        try:
            batch_size = z1.size(0)

            # 拼接特征 [2*N, D]
            features = torch.cat([z1, z2], dim=0)

            # 计算相似度矩阵 [2*N, 2*N]
            similarity_matrix = F.cosine_similarity(
                features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature

            # 创建对角线mask来排除自身相似度
            diag_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)

            # 构造正样本对的mask
            mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z1.device)
            mask[:batch_size, batch_size:] = torch.eye(batch_size)
            mask[batch_size:, :batch_size] = torch.eye(batch_size)

            # 应用diag_mask排除自身相似度
            similarity_matrix = similarity_matrix * diag_mask.float()

            # 计算损失 - 只使用正样本对和非自身的负样本
            positives = similarity_matrix[mask].view(2 * batch_size, 1)

            # 对于负样本,使用 ~mask & diag_mask 来同时排除正样本对和自身
            neg_mask = ~mask & diag_mask
            negatives = similarity_matrix[neg_mask].view(2 * batch_size, -1)

            # 拼接正负样本的相似度
            logits = torch.cat([positives, negatives], dim=1)

            # 创建标签 - 正样本在第一列
            labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)

            # 计算交叉熵损失
            loss = F.cross_entropy(logits, labels)

            return loss

        except Exception as e:
            logging.error(f"计算 InfoNCE 损失时出错: {str(e)}")
            logging.error(f"Input shapes - z1: {z1.shape}, z2: {z2.shape}")
            logging.error(f"Device - z1: {z1.device}, z2: {z2.device}")
            raise e

def train_simclr(
        args,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 200,
        device: str = 'cuda',
        learning_rate: float = 0.03,
        temperature: float = 0.1,
        weight_decay: float = 1e-4,
        logging_interval: int = 5,
        k: int = 4
) -> nn.Module:
    """SimCLR训练函数"""
    try:
        # 模型移到指定设备
        model = model.to(device)

        # 初始化优化器和调度器
        try:
            optimizer = SGD(
                model.parameters(),
                lr=args.simclr_lr,
                momentum=args.simclr_momentum,
                weight_decay=args.simclr_weight_decay
            )

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.pretrain_epochs,
                eta_min=learning_rate * 0.01
            )

            criterion = InfoNCELoss(temperature=temperature).to(device)
        except Exception as e:
            logging.error(f"初始化优化器、调度器或损失函数时出错: {str(e)}")
            raise e

        # 记录配置
        logging.info(f"\nSimCLR Pre-training Configuration:")
        logging.info(f"Epochs: {epochs}")
        logging.info(f"Learning rate: {learning_rate}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"Batch size: {train_loader.batch_size}")
        logging.info(f"KNN neighbors: {k}")

        best_loss = float('inf')
        best_model = None
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            try:
                model.train()
                total_loss = 0.0
                num_batches = len(train_loader)

                for batch_idx, ((x1, x2), _, _, _) in enumerate(train_loader):
                    try:
                        # 数据移到设备
                        x1, x2 = x1.to(device), x2.to(device)

                        # 优化器清零
                        optimizer.zero_grad()

                        # 前向传播
                        _, _, z1, z2 = model(x1, x2)

                        # 计算损失
                        loss = criterion(z1, z2)

                        # 反向传播
                        loss.backward()

                        # 梯度裁剪
                        try:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        except Exception as e:
                            logging.warning(f"梯度裁剪时出现警告: {str(e)}")

                        # 优化器步进
                        optimizer.step()

                        # 累积损失
                        total_loss += loss.item()

                        # 记录训练进度
                        if (batch_idx + 1) % logging_interval == 0:
                            avg_loss = total_loss / (batch_idx + 1)
                            logging.info(f'Epoch [{epoch + 1}/{epochs}] '
                                         f'Batch [{batch_idx + 1}/{num_batches}] '
                                         f'Loss: {avg_loss:.4f}')

                    except Exception as e:
                        logging.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                        continue  # 跳过这个批次,继续训练
                epoch_time = time.time() - epoch_start_time
                # 学习率调度
                scheduler.step()

                # 计算epoch平均损失
                avg_epoch_loss = total_loss / num_batches
                # 修改日志输出,添加时间信息
                logging.info(f'Epoch [{epoch + 1}/{epochs}] '
                             f'Average Loss: {avg_epoch_loss:.4f} '
                             f'Time: {epoch_time:.2f}s')

                # 早停检查
                if avg_epoch_loss < best_loss:
                    try:
                        best_loss = avg_epoch_loss
                        best_model = copy.deepcopy(model)
                        patience_counter = 0
                        logging.info(f"New best model saved with loss: {best_loss:.4f}")
                    except Exception as e:
                        logging.error(f"保存最佳模型时出错: {str(e)}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            except Exception as e:
                logging.error(f"训练epoch {epoch + 1} 时出错: {str(e)}")
                continue  # 跳过这个epoch,继续训练

        # 确保我们有最佳模型可以返回
        if best_model is None:
            logging.warning("没有找到最佳模型,返回最后的模型状态")
            best_model = model

        return best_model

    except Exception as e:
        logging.error(f"SimCLR训练过程发生严重错误: {str(e)}")
        raise e


def extract_features(model: nn.Module, train_loader: DataLoader, device: str) -> torch.Tensor:
    """提取特征
    Args:
        model: SimCLR模型
        train_loader: 数据加载器
        device: 设备
    Returns:
        torch.Tensor: 特征矩阵 [N, feature_dim]
    """
    model.eval()
    features = []

    with torch.no_grad():
        # 注意这里使用数据加载器的原始格式
        for (x, _), _, _, _ in train_loader:
            x = x.to(device)
            # 使用get_features方法提取特征
            feature = model.get_features(x)  # [batch_size, feature_dim]
            features.append(feature.cpu())

    return torch.cat(features, dim=0)  # [N, feature_dim]

def compute_knn_indices(
        features: torch.Tensor,
        k: int = 3
) -> np.ndarray:
    """计算KNN索引(仅使用欧氏距离)

    Args:
        features: 特征张量
        k: KNN邻居数量

    Returns:
        np.ndarray: KNN索引数组,shape为(n_samples, k)

    Raises:
        ValueError: 当k值大于样本数量减1时
        RuntimeError: 当特征转换或计算失败时
    """
    try:
        # 验证输入
        if not isinstance(features, torch.Tensor):
            raise TypeError("输入features必须是torch.Tensor类型")

        if not isinstance(k, int) or k < 1:
            raise ValueError("k必须是正整数")

        # 转换到CPU并转为numpy数组
        try:
            features_np = features.cpu().numpy()
        except Exception as e:
            logging.error(f"特征转换到numpy数组时出错: {str(e)}")
            raise RuntimeError("特征转换失败")

        n_samples = len(features_np)

        # 验证k值
        if k >= n_samples:
            raise ValueError(f"k值({k})必须小于样本数量({n_samples})")

        # 初始化距离矩阵
        try:
            dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

            # 计算欧氏距离矩阵
            for i in range(n_samples):
                try:
                    # 使用广播机制计算距离
                    diff = features_np - features_np[i].reshape(1, -1)
                    dist = np.linalg.norm(diff, axis=1)
                    dist_matrix[i] = dist
                except Exception as e:
                    logging.error(f"计算第{i}个样本的距离时出错: {str(e)}")
                    raise RuntimeError(f"距离计算失败: 样本 {i}")

        except Exception as e:
            logging.error(f"初始化或填充距离矩阵时出错: {str(e)}")
            raise RuntimeError("距离矩阵计算失败")

        # 找到最近的k个邻居
        try:
            knn_indices = []
            for i in range(n_samples):
                # 排除自身并获取k个最近邻
                sorted_indices = np.argsort(dist_matrix[i])
                # 跳过索引0(自身)，取接下来的k个索引
                indices = sorted_indices[1:k + 1]
                knn_indices.append(indices)

            knn_array = np.array(knn_indices)

            # 验证输出
            if knn_array.shape != (n_samples, k):
                raise ValueError(f"KNN索引数组形状错误: {knn_array.shape}, 期望: {(n_samples, k)}")

            return knn_array

        except Exception as e:
            logging.error(f"构建KNN索引数组时出错: {str(e)}")
            raise RuntimeError("KNN索引构建失败")

    except Exception as e:
        logging.error(f"KNN计算过程发生错误: {str(e)}")
        raise e