# utils/crda_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple
import logging
import numpy as np
import copy

class CRDALoss:
    def __init__(
            self,
            num_classes: int,
            feature_dim: int,
            device: str,
            beta_start: float = 0.95,
            beta_end: float = 0.7,
            alpha: float = 0.01,
            temperature: float = 0.07,
            contrast_threshold: float = 0.8,
            consistency_threshold: float = 0.7,
            lambda_contrast: float = 0.1,
            lambda_consistency: float = 0.1,
            late_start_epoch: int = 20,
            ema_decay: float = 0.999,
            fed_rounds: int = 100
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.alpha = alpha
        self.temperature = temperature
        self.contrast_threshold = contrast_threshold
        self.consistency_threshold = consistency_threshold
        
        # 保存基础权重值
        self.lambda_contrast = lambda_contrast
        self.lambda_consistency = lambda_consistency
        
        self.late_start_epoch = late_start_epoch
        self.ema_decay = ema_decay
        self.fed_rounds = fed_rounds
        self.current_round = 0
        self.ema_model = None
        
        self._log_initialization()
        
    def get_adaptive_weights(self, confidence: torch.Tensor) -> Tuple[float, float]:
        """
        计算自适应权重
        Args:
            confidence: 样本的预测置信度 [batch_size]
        Returns:
            lambda_contrast, lambda_consistency: 对比损失和一致性损失的权重
        """
        # 基于轮次的线性增加
        progress = min(self.current_round / self.late_start_epoch, 1.0)
        
        # 基于置信度的权重调整
        conf_mean = torch.mean(confidence).item()
        
        lambda_contrast = self.lambda_contrast_base * progress * conf_mean
        lambda_consistency = self.lambda_consistency_base * progress * conf_mean
        
        return lambda_contrast, lambda_consistency
    def _log_initialization(self):
        """记录初始化信息"""
        logging.info("\nInitializing C-RDA Loss:")
        logging.info(f"Number of classes: {self.num_classes}")
        logging.info(f"Feature dimension: {self.feature_dim}")
        logging.info(f"Beta range: [{self.beta_end}, {self.beta_start}]")
        logging.info(f"Alpha: {self.alpha}")
        logging.info(f"Late start epoch: {self.late_start_epoch}")
        logging.info(f"Using Mixup: {self.use_mixup}")
        if self.use_mixup:
            logging.info(f"Mixup alpha: {self.mixup_alpha}")

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """仅在use_mixup=True时使用"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, outputs: torch.Tensor, targets_a: torch.Tensor,
                        targets_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Mixup loss"""
        # 计算RDA loss而不是交叉熵
        loss_a = self.calculate_credal_loss(outputs, targets_a)
        loss_b = self.calculate_credal_loss(outputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b

    def update_ema_model(self, model: nn.Module):
        """更新EMA模型"""
        if self.ema_model is None:
            # 创建新模型并复制状态字典
            self.ema_model = type(model)(num_classes=self.num_classes).to(self.device)
            self.ema_model.load_state_dict(model.state_dict())
        else:
            # 使用状态字典更新
            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), self.ema_model.parameters()):
                    param_k.data = param_k.data * self.ema_decay + param_q.data * (1 - self.ema_decay)

    def get_dynamic_beta(self) -> float:
        """计算动态beta值"""
        T = self.current_round
        T_max = self.fed_rounds
        beta_t = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (
                1 + math.cos(min(T / T_max, 1.0) * math.pi)
        )
        return beta_t

    def calculate_credal_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算RDA的credal loss"""
        batch_size = outputs.size(0)

        # 计算预测概率和log概率
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)

        # 获取当前beta值
        beta = self.get_dynamic_beta()

        # 构建可能性分布π [batch_size, num_classes]
        possibility = torch.ones((batch_size, self.num_classes)).to(self.device) * self.alpha

        # 设置原始标签的可能性为1
        possibility[torch.arange(batch_size), targets] = 1.0

        # 设置高置信度预测的可能性为1
        high_conf_mask = probs >= beta
        possibility[high_conf_mask] = 1.0

        # 计算投影分布p_r
        p_r = torch.zeros_like(probs)

        # 处理π(y)=1的部分
        full_conf_mask = (possibility == 1.0)
        if full_conf_mask.any():
            norm_term = torch.sum(probs * full_conf_mask.float(), dim=1, keepdim=True)
            norm_term = torch.clamp(norm_term, min=1e-12)
            p_r[full_conf_mask] = ((1 - self.alpha) *
                                   (probs * full_conf_mask.float()) / norm_term)[full_conf_mask]

        # 处理π(y)=α的部分
        low_conf_mask = (possibility == self.alpha)
        if low_conf_mask.any():
            norm_term = torch.sum(probs * low_conf_mask.float(), dim=1, keepdim=True)
            norm_term = torch.clamp(norm_term, min=1e-12)
            p_r[low_conf_mask] = (self.alpha *
                                  (probs * low_conf_mask.float()) / norm_term)[low_conf_mask]

        # 计算KL散度损失
        kl_div = torch.sum(p_r * (torch.log(torch.clamp(p_r, min=1e-12)) - log_probs), dim=1)

        return kl_div.mean()

    def calculate_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor,
                                   confidence: torch.Tensor) -> torch.Tensor:
        """计算对比学习损失"""
        if self.current_round < self.late_start_epoch:
            return torch.tensor(0.0).to(self.device)

        # 只对高置信度样本计算对比损失
        high_conf_mask = confidence >= self.contrast_threshold
        if not high_conf_mask.any():
            return torch.tensor(0.0).to(self.device)

        features = F.normalize(features[high_conf_mask], dim=1)
        labels = labels[high_conf_mask]

        # 检查是否有足够的样本
        if features.size(0) < 2:
            return torch.tensor(0.0).to(self.device)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)

        # 创建标签匹配矩阵
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        # 移除对角线元素
        mask = torch.eye(label_matrix.size(0), dtype=torch.bool).to(self.device)
        label_matrix = label_matrix[~mask].view(label_matrix.size(0), -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)

        # 检查是否存在正样本对
        if not label_matrix.any():
            return torch.tensor(0.0).to(self.device)

        # 计算正样本对的损失
        positives = similarity_matrix[label_matrix].view(-1, 1)
        negatives = similarity_matrix[~label_matrix].view(positives.size(0), -1)

        # 确保正负样本都存在
        if positives.size(0) == 0 or negatives.size(1) == 0:
            return torch.tensor(0.0).to(self.device)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        return F.cross_entropy(logits, labels)
    
    
    
    def calculate_consistency_loss(
            self,
            model: nn.Module,
            outputs: torch.Tensor,
            confidence: torch.Tensor,
            inputs: torch.Tensor = None  # 添加inputs参数
    ) -> torch.Tensor:
        """计算一致性损失"""
        if self.current_round < self.late_start_epoch or inputs is None:
            return torch.tensor(0.0).to(self.device)

        # 只对中等置信度以上的样本计算一致性损失
        conf_mask = confidence >= self.consistency_threshold
        if not conf_mask.any():
            return torch.tensor(0.0).to(self.device)

        # 更新和使用EMA模型
        self.update_ema_model(model)

        # EMA模型预测
        with torch.no_grad():
            self.ema_model.eval()
            ema_outputs = self.ema_model(inputs)

        outputs = outputs[conf_mask]
        ema_outputs = ema_outputs[conf_mask]

        # 使用KL散度计算一致性损失
        p1 = F.softmax(outputs, dim=1)
        p2 = F.softmax(ema_outputs, dim=1)

        consistency_loss = (
                                   F.kl_div(p1.log(), p2, reduction='batchmean') +
                                   F.kl_div(p2.log(), p1, reduction='batchmean')
                           ) / 2.0

        return consistency_loss

    def __call__(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        # 1. 前期策略 - 仅RDA
        if self.current_round < self.late_start_epoch:
            loss = self.calculate_credal_loss(outputs, targets)
            return {
                'total_loss': loss,
                'credal_loss': loss,
                'contrast_loss': torch.tensor(0.0).to(self.device),
                'consistency_loss': torch.tensor(0.0).to(self.device)
            }
        
        # 2. 后期策略(RDA + 对比 + 一致性)
        credal_loss = self.calculate_credal_loss(outputs, targets)
        
        if features is None:
            features = model.get_features(inputs)
        
        probs = F.softmax(outputs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        
        contrast_loss = self.calculate_contrastive_loss(features, targets, confidence)
        consistency_loss = self.calculate_consistency_loss(
            model=model,
            outputs=outputs, 
            confidence=confidence,
            inputs=inputs
        )
        
        lambda_contrast, lambda_consistency = self.get_adaptive_weights(confidence)
        
        total_loss = (
            credal_loss + 
            lambda_contrast * contrast_loss +
            lambda_consistency * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'credal_loss': credal_loss,
            'contrast_loss': contrast_loss,
            'consistency_loss': consistency_loss,
            'lambda_contrast': lambda_contrast,
            'lambda_consistency': lambda_consistency
        }

    def update_round(self, round_num: int):
        """更新当前轮次"""
        self.current_round = round_num


def test_consistency_loss():
    """测试一致性损失计算"""
    import torch.nn as nn

    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc = nn.Linear(10, num_classes)

        def forward(self, x):
            return self.fc(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    batch_size = 4
    feature_dim = 10

    # 初始化CRDALoss
    crda_loss = CRDALoss(
        num_classes=num_classes,
        feature_dim=feature_dim,
        device=device,
        late_start_epoch=2  # 设置小一点便于测试
    )
    crda_loss.current_round = 3  # 设置为后期轮次

    # 创建测试数据
    model = SimpleModel(num_classes).to(device)
    inputs = torch.randn(batch_size, feature_dim).to(device)
    outputs = model(inputs)
    confidence = torch.rand(batch_size).to(device)

    try:
        # 测试一致性损失计算
        consistency_loss = crda_loss.calculate_consistency_loss(
            model=model,
            outputs=outputs,
            confidence=confidence,
            inputs=inputs
        )
        print("一致性损失计算成功!")
        print(f"Loss value: {consistency_loss.item()}")

        # 测试EMA模型更新
        if crda_loss.ema_model is not None:
            print("EMA模型创建成功!")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_consistency_loss()
    if success:
        print("所有测试通过!")