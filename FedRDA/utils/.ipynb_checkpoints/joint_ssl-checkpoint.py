# utils/joint_ssl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple


class ContrastLearner:
    """对比学习训练逻辑,实现论文中的对比学习方法"""

    def __init__(
            self,
            momentum: float = 0.99,  # 动量系数
            tau: float = 0.8,  # 对比阈值
            lambda_ctr: float = 50.0  # 对比损失权重
    ):
        self.momentum = momentum
        self.tau = tau
        self.lambda_ctr = lambda_ctr

    def compute_contrast_loss(
            self,
            p1: torch.Tensor, 
            z2: torch.Tensor,
            outputs: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """计算对比损失(修正版)
        Args:
            p1: 第一个视图的预测 [N, dim]
            z2: 第二个视图的特征 [N, dim] 
            outputs: 分类输出 [N, num_classes]
            batch_size: 批大小
        """
        # 1. 确保输入在合理范围内
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)
    
        # 2. 计算相似度矩阵 
        contrast = torch.matmul(p1, z2.t())  # [N, N]
    
        # 3. 计算对比logits
        # -<q,z> + log(1-<q,z>)部分
        contrast_1 = -contrast * torch.zeros(batch_size, batch_size).fill_diagonal_(1).cuda() + \
                    ((1 - contrast).log()) * torch.ones(batch_size, batch_size).fill_diagonal_(0).cuda()
        
        # 添加常数项2
        contrast_logits = 2 + contrast_1
    
        # 4. 计算概率掩码
        soft_targets = torch.softmax(outputs, dim=1)
        contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
        contrast_mask.fill_diagonal_(1)
    
        # 5. 应用阈值
        pos_mask = (contrast_mask >= self.tau).float()
        contrast_mask = contrast_mask * pos_mask
        
        # 6. 归一化掩码
        # 添加数值稳定性
        mask_sum = contrast_mask.sum(1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-12)  # 避免除零
        contrast_mask = contrast_mask / mask_sum
    
        # 7. 计算最终损失
        loss = (contrast_logits * contrast_mask).sum(dim=1).mean(0)
    
        return loss

    def train_step(
            self,
            model: nn.Module,
            batch_data: Tuple,  # 修改参数名和类型
            optimizer: torch.optim.Optimizer,
            device: str
    ) -> Dict:
        """对比学习训练步骤
        Args:
            model: ResNetModel实例
            images: (im_aug1, im_aug2, im_weak)三个视图
            targets: 标签
            optimizer: 优化器
            device: 设备
        """
        model.train()
        (images, targets, indices, original_targets) = batch_data
        im_aug1, im_aug2, im_weak = [img.to(device) for img in images]
        targets = targets.to(device)
        batch_size = targets.size(0)

        # 对比学习分支
        p1, z2 = model((im_aug1, im_aug2), mode='contrast', m=self.momentum)

        # 分类分支
        outputs = model(im_weak, mode='train')

        # 计算损失
        loss_ctr = self.compute_contrast_loss(p1, z2, outputs, batch_size)
        loss_ce = F.cross_entropy(outputs, targets)

        # 总损失
        loss = self.lambda_ctr * loss_ctr + loss_ce

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            accuracy = 100.0 * correct / batch_size

        # 返回训练指标
        return {
            'loss': loss.item(),
            'contrast_loss': loss_ctr.item(),
            'ce_loss': loss_ce.item(),
            'accuracy': accuracy,
            'batch_size': batch_size
        }

    def train_epoch(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            device: str
    ) -> Tuple[float, Dict]:
        """训练一个epoch
        Args:
            model: ResNetModel实例
            train_loader: 训练数据加载器(提供三视图数据)
            optimizer: 优化器
            device: 设备
        Returns:
            avg_loss: 平均损失
            metrics: 训练指标字典
        """
        total_loss = 0.0
        total_ctr_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # 训练一步
            metrics = self.train_step(
                model=model,
                batch_data=batch_data,
                optimizer=optimizer,
                device=device
            )

            # 更新统计信息
            total_loss += metrics['loss'] * metrics['batch_size']
            total_ctr_loss += metrics['contrast_loss'] * metrics['batch_size']
            total_ce_loss += metrics['ce_loss'] * metrics['batch_size']
            total_correct += (metrics['accuracy'] / 100.0) * metrics['batch_size']
            total_samples += metrics['batch_size']

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {metrics["loss"]:.4f}, '
                    f'Contrast Loss: {metrics["contrast_loss"]:.4f}, '
                    f'CE Loss: {metrics["ce_loss"]:.4f}, '
                    f'Accuracy: {metrics["accuracy"]:.2f}%'
                )

        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_metrics = {
            'train_loss': avg_loss,
            'train_contrast_loss': total_ctr_loss / total_samples,
            'train_ce_loss': total_ce_loss / total_samples,
            'train_accuracy': 100.0 * total_correct / total_samples
        }

        return avg_loss, avg_metrics

    def evaluate(
            self,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            device: str
    ) -> Dict:
        """评估模型
        Args:
            model: ResNetModel实例
            val_loader: 验证数据加载器
            device: 设备
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                # 使用测试模式
                outputs = model(images, mode='test')
                loss = F.cross_entropy(outputs, targets)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

        return {
            'val_loss': total_loss / total_samples,
            'val_accuracy': 100.0 * total_correct / total_samples
        }