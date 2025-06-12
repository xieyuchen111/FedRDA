# utils/trainer.py
import torch
import torch.nn.functional as F
import logging
from typing import Dict
import math


class SemiSupervisedTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            # criterion: torch.nn.Module,
            device: str,
            num_classes: int,
            loss_type: str = 'rda',
            beta_start: float = 0.9,  # 初始置信度阈值
            beta_end: float = 0.8,  # 最终置信度阈值
            alpha: float = 0.1,  # 标签松弛参数
            local_epochs: int = 5,
            fed_rounds: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.loss_type = loss_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.alpha = alpha
        self.fed_rounds = fed_rounds
        self.current_round = 0
        self.current_local_epoch = 0
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(device) 
        self._log_initialization()
        # 添加噪声检测相关属性
        self.soft_labels = None
        self.noisy_indices = []
        self.noise_detector = None
        
    def _log_initialization(self):
        """记录初始化信息"""
        logging.info(f"\nInitializing RDA Trainer:")
        logging.info(f"Number of classes: {self.num_classes}")
        logging.info(f"Beta range: [{self.beta_end}, {self.beta_start}]")
        logging.info(f"Alpha: {self.alpha}")
        logging.info(f"Local epochs: {self.local_epochs}")
    
    def get_dynamic_beta(self) -> float:
        """计算动态beta值"""
        T = self.current_round
        T_max = self.fed_rounds
        beta_t = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (
                1 + math.cos(min(T / T_max, 1.0) * math.pi)
        )
        # if self.current_local_epoch % 10 == 0:
        #     # logging.info(f"Current beta: {beta_t:.4f}")
        return beta_t
        
    def calculate_gce_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算广义交叉熵损失 L_gce = (1 - (p_y)^q) / q
            Args:
                outputs: 模型输出 shape [batch_size, num_classes]
                targets: 真实标签 shape [batch_size]
            Returns:
                loss: 标量损失值
            """
        # 获取预测概率分布 
        probs = F.softmax(outputs, dim=1)  # [batch_size, num_classes]
    
        # 获取每个样本对应真实标签的预测概率
        # probs[i][targets[i]] 表示第i个样本真实类别的预测概率
        target_probs = probs[torch.arange(probs.size(0)), targets]  # [batch_size]
    
        # 计算GCE损失 L = (1 - p^q) / q
        q = 0.7  # 论文推荐的参数值
        loss = (1 - torch.pow(target_probs + 1e-10, q)) / q  # 加上1e-10避免数值问题
    
        # 返回batch的平均损失
        return loss.mean()

    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失函数"""
        if self.loss_type == 'ce':
            return self.ce_criterion(outputs, targets)
        elif self.loss_type == 'gce':
            return self.calculate_gce_loss(outputs, targets)
        else:  # rda
            return self.calculate_credal_loss(outputs, targets)
        
    def calculate_credal_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算credal loss,使用矩阵运算
        Args:
            outputs: [batch_size, num_classes]
            targets: [batch_size]
        """
        batch_size = outputs.size(0)

        # 计算预测概率和log概率
        probs = F.softmax(outputs, dim=1)  # [batch_size, num_classes]
        log_probs = F.log_softmax(outputs, dim=1)  # [batch_size, num_classes]

        # 获取当前beta值
        beta = self.get_dynamic_beta()

        # 构建可能性分布π [batch_size, num_classes]
        possibility = torch.ones((batch_size, self.num_classes)).to(self.device) * self.alpha

        # 设置原始标签的可能性为1
        possibility[torch.arange(batch_size), targets] = 1.0

        # 设置高置信度预测的可能性为1
        high_conf_mask = probs >= beta
        possibility[high_conf_mask] = 1.0

        # 计算投影分布p_r [batch_size, num_classes]
        p_r = torch.zeros_like(probs)

        # 处理π(y)=1的部分
        full_conf_mask = (possibility == 1.0)
        if full_conf_mask.any():
            # 计算归一化项 [batch_size, 1]
            norm_term = torch.sum(probs * full_conf_mask.float(), dim=1, keepdim=True)
            norm_term = torch.clamp(norm_term, min=1e-12)  # 数值稳定性
            # 计算投影 [batch_size, num_classes]
            p_r[full_conf_mask] = ((1 - self.alpha) *
                                   (probs * full_conf_mask.float()) / norm_term)[full_conf_mask]

        # 处理π(y)=α的部分
        low_conf_mask = (possibility == self.alpha)
        if low_conf_mask.any():
            # 计算归一化项 [batch_size, 1]
            norm_term = torch.sum(probs * low_conf_mask.float(), dim=1, keepdim=True)
            norm_term = torch.clamp(norm_term, min=1e-12)  # 数值稳定性
            # 计算投影 [batch_size, num_classes]
            p_r[low_conf_mask] = (self.alpha *
                                  (probs * low_conf_mask.float()) / norm_term)[low_conf_mask]

        # 计算KL散度损失 [batch_size]
        kl_div = torch.sum(p_r * (torch.log(torch.clamp(p_r, min=1e-12)) - log_probs), dim=1)

        # 返回平均损失
        loss = kl_div.mean()

        if self.current_local_epoch % 10 == 0:
            with torch.no_grad():
                num_in_credal = torch.sum((kl_div < 1e-6).float()).item()
                # logging.info(f"Samples in credal set: {num_in_credal}/{batch_size}")

        return loss

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor) -> Dict:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(inputs, mode='train')  # [batch_size, num_classes]
        features = self.model.get_features(inputs)

        # 初始化损失
        batch_size = targets.size(0)
        losses = torch.zeros(batch_size, device=self.device)

        # 初始化有效样本掩码 - 这是新增的
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # 处理干净样本和噪声样本
        clean_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if self.soft_labels and self.noisy_indices:
            for i, idx in enumerate(indices):
                if idx.item() in self.noisy_indices:
                    clean_mask[i] = False
                    # 检查软标签是否有效 - 这是修改的部分
                    soft_label_info = self.soft_labels[idx.item()]
                    if soft_label_info['soft_label'] is None:
                        valid_mask[i] = False  # 标记为无效样本，不参与训练

        # 计算干净样本的损失
        if clean_mask.any():
            clean_losses = F.cross_entropy(outputs[clean_mask], targets[clean_mask], reduction='none')
            losses[clean_mask] = clean_losses

        # 处理噪声样本（只处理有效软标签的样本）
        if self.soft_labels and self.noisy_indices:
            noisy_mask = ~clean_mask
            for i, idx in enumerate(indices):
                if idx.item() in self.noisy_indices and valid_mask[i]:  # 只处理有效的噪声样本
                    soft_label_info = self.soft_labels[idx.item()]
                    if soft_label_info['soft_label'] is not None:
                        soft_target = torch.tensor(soft_label_info['soft_label']).to(self.device)
                        probs = F.log_softmax(outputs[i], dim=0)
                        losses[i] = -torch.sum(soft_target * probs)

        # 只对有效样本计算平均损失 - 这是修改的部分
        valid_losses = losses[valid_mask]
        if len(valid_losses) > 0:  # 确保有有效样本
            loss = valid_losses.mean()
            loss.backward()
            self.optimizer.step()

        # 更新噪声检测器历史记录
        if self.noise_detector:
            self.noise_detector.update_history(
                indices=indices[valid_mask],  # 只记录有效样本
                losses=losses[valid_mask],
                targets=targets[valid_mask],
                epoch=self.current_local_epoch,
                features=features[valid_mask],
                logits=outputs[valid_mask]
            )

        # 计算准确率（只考虑有效样本）
        with torch.no_grad():
            _, predicted = torch.max(outputs[valid_mask].data, 1)
            correct = (predicted == targets[valid_mask]).sum().item()
            total = valid_mask.sum().item()
            accuracy = 100.0 * correct / total if total > 0 else 0.0

        skipped_samples = batch_size - total
  
        return {
            'loss': loss.item() if total > 0 else 0.0,
            'accuracy': accuracy,
            'batch_size': total,
            # 'skipped_samples': skipped_samples  # 新增：记录跳过的样本数
        }
        
    def train_epoch(self, train_loader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets, indices, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            indices = indices.to(self.device)

            batch_metrics = self.train_step(inputs, targets, indices)

            total_loss += batch_metrics['loss'] * batch_metrics['batch_size']
            total_correct += (batch_metrics['accuracy'] / 100.0) * batch_metrics['batch_size']
            total_samples += batch_metrics['batch_size']

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / total_samples
                cur_acc = 100.0 * total_correct / total_samples
                logging.info(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                            f'Loss: {avg_loss:.4f}, '
                            f'Acc: {cur_acc:.2f}%')

        return {
            'train_loss': total_loss / total_samples,
            'train_accuracy': 100.0 * total_correct / total_samples
        }

    def evaluate(self, val_loader) -> Dict:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets, indices, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs, mode='test') 
                loss = self.calculate_credal_loss(outputs, targets)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
 
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def train(self, args, train_loader, val_loader) -> Dict:
        """完整训练过程"""
        logging.info(f"\nStarting training for {self.local_epochs} epochs")
        logging.info(f"Current round: {self.current_round}")

        best_val_acc = 0.0
        train_metrics = None
        val_metrics = None

        for epoch in range(1, self.local_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.current_round += 1

            logging.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Train Acc={train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val Acc={val_metrics['val_accuracy']:.2f}%"
            )

            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']

        return {
            'train_loss': train_metrics['train_loss'],
            'train_accuracy': train_metrics['train_accuracy'],
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['val_accuracy']
        }