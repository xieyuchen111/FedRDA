# federated/client.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import logging
import time
from typing import Dict, List
from utils.client_detector import NoiseClientDetector 
from utils.trainer import SemiSupervisedTrainer
from utils.noise_detection import DynamicNoiseDetector

class FederatedClient:
    def __init__(
            self,
            args,
            client_id: int,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: str,
            is_noisy: bool = False,
            noise_ratio: float = 0.0,
            learning_rate: float = 0.001,
            local_epochs: int = 5,
            loss_type: str = 'ce',
            num_classes: int = 2880,
            # 添加不确定性训练参数
            window_size: int = 5,
            tau: float = 0.6,
            min_samples: int = 10,
            alpha:int=0.1
    ):
        """联邦学习客户端

        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            is_noisy: 是否为噪声客户端
            noise_ratio: 噪声比例
            learning_rate: 学习率
            local_epochs: 本地训练轮数
            num_classes: 类别总数
            window_size: 损失曲线窗口大小
            tau: GMM阈值
            min_samples: GMM拟合最小样本数
        """
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_noisy = is_noisy
        self.noise_ratio = noise_ratio
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_classes = num_classes

        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.fed_lr,  # 建议初始学习率设为 1e-3
            weight_decay=args.fed_weight_decay,  # 建议设为 0.01
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.trainer = SemiSupervisedTrainer(
            model=self.model,
            optimizer=self.optimizer,
            # criterion=self.criterion,
            device=self.device,
            loss_type=args.loss_type,
            num_classes=self.num_classes,
            local_epochs=self.local_epochs,
            fed_rounds=args.fed_rounds
        )
        # 初始化噪声检测器
        self.noise_detector = DynamicNoiseDetector(
            num_classes=self.num_classes,
            detection_round=args.detection_start_round,  # 从args获取检测轮次
            noise_ratio=self.noise_ratio,
            feature_dim=512,  # ResNet的特征维度
            temperature=0.5,  # 软标签温度参数
            confidence_threshold=0.9,  # 预测置信度阈值
            device=self.device  # 设备
        )

        # 记录客户端配置
        self._log_client_config()

    def set_pretrain_info(self, features, knn_indices):
        """设置预训练的特征和KNN信息"""
        self.pretrain_features = features
        self.pretrain_knn = knn_indices

    def _log_client_config(self):
        """记录客户端配置信息"""
        logging.info(f"\nClient {self.client_id} Configuration:")
        logging.info(f"Noisy Client: {self.is_noisy}")
        if self.is_noisy:
            logging.info(f"Noise Ratio: {self.noise_ratio}")
        logging.info(f"Number of Classes: {self.num_classes}")
        logging.info(f"Training Samples: {len(self.train_loader.dataset)}")
        logging.info(f"Validation Samples: {len(self.val_loader.dataset)}")
        # logging.info(f"Detection Begin Epoch: {self.detection_begin_epoch}")
        # logging.info(f"Late Stage Epoch: {self.late_stage_epoch}")
        # logging.info(f"Confidence Threshold: {self.confidence_threshold}")
        logging.info(f"Local Epochs: {self.local_epochs}")
        logging.info(f"Learning Rate: {self.learning_rate}")

    def _add_label_noise(self):
        """为训练数据添加标签噪声"""
        dataset = self.train_loader.dataset

        # 获取当前客户端的类别数量
        unique_classes = set(dataset.targets)
        num_classes = len(unique_classes)

        # 计算需要添加噪声的样本数
        num_samples = len(dataset)
        num_noisy = int(self.noise_ratio * num_samples)

        # 保存原始标签
        self.original_targets = copy.deepcopy(dataset.targets)

        # 随机选择要添加噪声的索引
        noisy_indices = np.random.choice(range(num_samples), num_noisy, replace=False)

        # 记录每个类别的噪声样本数量
        noise_per_class = {cls: 0 for cls in unique_classes}

        # 添加噪声
        for idx in noisy_indices:
            current_label = dataset.targets[idx]
            # 随机选择一个不同的标签
            possible_labels = list(range(num_classes))
            possible_labels.remove(current_label)
            new_label = np.random.choice(possible_labels)
            dataset.targets[idx] = new_label
            noise_per_class[current_label] += 1

        # 记录噪声添加情况
        logging.info(f"\nClient {self.client_id} Noise Addition Details:")
        logging.info(f"Total samples: {num_samples}")
        logging.info(f"Noisy samples: {num_noisy}")
        logging.info("Noise distribution per class:")
        for cls, count in noise_per_class.items():
            if count > 0:
                logging.info(f"Class {cls}: {count} noisy samples")


    def train(self, args, current_round: int, is_detected_noisy: bool = False) -> Dict:
        """本地训练，每轮输出基础指标"""
        logging.info(f"\nTraining Client {self.client_id}, Round {current_round}")
        logging.info(f"Client status: {'Noisy(detected)' if is_detected_noisy else 'Clean'}")
        
        try:
            self.current_round = current_round
            self.noise_detector.current_round = current_round
            train_metrics = {}
            
            for local_epoch in range(1, self.local_epochs + 1):
                epoch_start_time = time.time()
                self.trainer.current_round = current_round
                self.trainer.current_local_epoch = local_epoch
                
                soft_labels = None
                noisy_indices = []
                if is_detected_noisy:
                    soft_labels, noisy_indices = self.noise_detector.detect_noisy_samples(
                        is_noisy_client=self.is_noisy,
                        model=self.model  # 传入模型用于评估
                    )
                    if noisy_indices:
                        logging.info(f"Detected {len(noisy_indices)} noisy samples")
                
                # 设置trainer的软标签信息
                self.trainer.soft_labels = soft_labels
                self.trainer.noisy_indices = noisy_indices
                self.trainer.noise_detector = self.noise_detector
    
                # 使用trainer进行训练
                epoch_metrics = self.trainer.train_epoch(self.train_loader)
                val_metrics = self.evaluate()
    
                epoch_metrics.update(val_metrics)
                if noisy_indices is None:
                    noisy_indices = []
                if is_detected_noisy and soft_labels:
                    clean_samples = len(self.train_loader.dataset) - len(noisy_indices)
                    epoch_metrics.update({
                        'noisy_samples': len(noisy_indices),
                        'clean_samples': clean_samples,
                        'noise_ratio': len(noisy_indices) / len(self.train_loader.dataset)
                    })
    
                epoch_time = time.time() - epoch_start_time
                
                # Log results
                logging.info(f"\nLocal Epoch {local_epoch}/{self.local_epochs} Results:")
                logging.info(f"Training Loss: {epoch_metrics['train_loss']:.4f}")
                logging.info(f"Training Accuracy: {epoch_metrics['train_accuracy']:.2f}%")
                if is_detected_noisy and soft_labels:
                    logging.info(f"Clean Samples: {epoch_metrics['clean_samples']}")
                    logging.info(f"Noisy Samples: {epoch_metrics['noisy_samples']}")
                    logging.info(f"Noise Ratio: {epoch_metrics['noise_ratio']:.2%}")
                logging.info(f"Validation Loss: {epoch_metrics['val_loss']:.4f}")
                logging.info(f"Validation Accuracy: {epoch_metrics['val_accuracy']:.2f}%")
                logging.info(f"Epoch Time: {epoch_time:.2f}s")
    
            # Save final metrics
            train_metrics = {
                'client_id': self.client_id,
                'is_noisy': is_detected_noisy,
                'train_loss': epoch_metrics['train_loss'],
                'train_accuracy': epoch_metrics['train_accuracy'],
                'val_loss': epoch_metrics['val_loss'],
                'val_accuracy': epoch_metrics['val_accuracy'],
                'num_samples': len(self.train_loader.dataset),
                'training_time': epoch_time
            }
            
            if is_detected_noisy and soft_labels:
                train_metrics.update({
                    'noisy_samples': epoch_metrics['noisy_samples'],
                    'clean_samples': epoch_metrics['clean_samples'],
                    'noise_ratio': epoch_metrics['noise_ratio']
                })
        
            return train_metrics
    
        except Exception as e:
            logging.error(f"Error training client {self.client_id}: {str(e)}")
            logging.error(traceback.format_exc())
            raise e

    def _log_training_results(self, metrics: Dict):
        """记录训练结果"""
        logging.info(f"\nClient {self.client_id} Training Results:")
        logging.info(f"Training Loss: {metrics['train_loss']:.4f}")
        logging.info(f"Training Accuracy: {metrics['train_accuracy']:.2f}%")
        logging.info(f"Validation Loss: {metrics['val_loss']:.4f}")
        logging.info(f"Validation Accuracy: {metrics['val_accuracy']:.2f}%")
        if self.is_noisy:
            logging.info(f"Client Type: Noisy (Noise Ratio: {self.noise_ratio})")
        else:
            logging.info(f"Client Type: Clean")
    def evaluate(self) -> Dict:
        """评估模型"""
        val_metrics = self.trainer.evaluate(self.val_loader)
        val_metrics['client_id'] = self.client_id
        val_metrics['is_noisy_client'] = self.is_noisy
        return val_metrics

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """设置模型参数"""
        self.model.load_state_dict(parameters)

    def restore_clean_labels(self):
        """恢复原始干净标签"""
        if self.is_noisy and hasattr(self, 'original_targets'):
            dataset = self.train_loader.dataset

            if isinstance(dataset, torch.utils.data.Subset):
                full_dataset = dataset.dataset
                subset_indices = dataset.indices

                for original_idx, subset_idx in enumerate(subset_indices):
                    full_dataset.targets[subset_idx] = self.original_targets[original_idx]
            else:
                dataset.targets = copy.deepcopy(self.original_targets)

            logging.info(f"Client {self.client_id}: Restored original clean labels")

