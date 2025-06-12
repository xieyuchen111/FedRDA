# utils/utils.py

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from typing import List, Tuple, Dict
import logging
import os
from datetime import datetime


class MappedDataset(Dataset):
    """封装数据集以支持噪声监测"""

    def __init__(self, dataset, indices, new_targets, class_mapping, original_targets=None):
        """
        Args:
            dataset: 原始数据集
            indices: 样本索引
            new_targets: 新标签
            class_mapping: 类别映射
            original_targets: 原始标签（用于噪声监测）
        """
        self.dataset = dataset
        self.indices = indices
        self.targets = new_targets
        self.class_mapping = class_mapping

        # 使用 numpy.copy 复制原始标签
        if original_targets is None:
            self.original_targets = np.copy(new_targets)
        else:
            self.original_targets = original_targets

    def __getitem__(self, idx):
        """返回(image, target, index, original_target)格式的数据"""
        image = self.dataset[self.indices[idx]][0]
        target = self.targets[idx]
        original_target = self.original_targets[idx]
        return image, target, self.indices[idx], original_target

    def __len__(self):
        return len(self.indices)

    def get_original_class(self, mapped_target):
        """获取原始类别"""
        inverse_mapping = {v: k for k, v in self.class_mapping.items()}
        return inverse_mapping[mapped_target]


def create_client_data_loaders(
        train_dataset,
        val_dataset,  # 可以为None，将使用train_dataset分割
        num_clients: int,
        batch_size: int,
        num_workers: int = 4,
        noise_ratio: float = 0.0,  # 添加噪声比例参数
        noisy_client_ids: List[int] = None,
        seed: int = 42, # 指定噪声客户端
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    为每个客户端创建数据加载器

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集(可选)
        num_clients: 客户端数量
        batch_size: 批次大小
        num_workers: 数据加载线程数
        noise_ratio: 噪声比例
        noisy_client_ids: 噪声客户端ID列表

    Returns:
        List[Tuple[DataLoader, DataLoader]]: 客户端数据加载器列表
    """
    # 获取每个类别的样本索引
    class_indices = {}
    for idx, target in enumerate(train_dataset.targets):
        if target not in class_indices:
            class_indices[target] = []
        if len(class_indices[target]) < 6:  # 每个类限制6个样本
            class_indices[target].append(idx)

    # 验证类别数量
    for cls, indices in class_indices.items():
        assert len(indices) == 6, f"Class {cls} has {len(indices)} samples, expected 6"

    # 计算每个客户端的类别数
    num_classes = len(class_indices)
    classes_per_client = num_classes // num_clients
    assert num_classes % num_clients == 0, "类别数必须能被客户端数整除"

    logging.info(f"Dataset Statistics:")
    logging.info(f"Total classes: {num_classes}")
    logging.info(f"Classes per client: {classes_per_client}")
    logging.info(f"Number of noisy clients: {len(noisy_client_ids) if noisy_client_ids else 0}")

    np.random.seed(seed)
    # 随机打乱类别分配
    class_list = list(class_indices.keys())
    np.random.shuffle(class_list)

    # 为每个客户端分配数据
    client_loaders = []
    current_class_idx = 0

    for client_idx in range(num_clients):
        # 获取该客户端的类别
        client_classes = class_list[current_class_idx:current_class_idx + classes_per_client]
        class_mapping = {old_class: new_idx for new_idx, old_class in enumerate(sorted(client_classes))}

        # 收集训练和验证样本
        train_indices = []
        train_targets = []
        val_indices = []
        val_targets = []
        original_train_targets = []
        original_val_targets = []

        for class_label in client_classes:
            samples = class_indices[class_label]
            # 前5个样本用于训练
            train_samples = samples[:5]
            train_indices.extend(train_samples)
            mapped_target = class_mapping[class_label]
            train_targets.extend([mapped_target] * len(train_samples))
            original_train_targets.extend([mapped_target] * len(train_samples))

            # 最后1个样本用于验证
            val_samples = samples[5:]
            val_indices.extend(val_samples)
            val_targets.extend([mapped_target] * len(val_samples))
            original_val_targets.extend([mapped_target] * len(val_samples))

        # 如果是噪声客户端，添加标签噪声
        is_noisy = noisy_client_ids and client_idx in noisy_client_ids
        if is_noisy and noise_ratio > 0:
            num_noisy = int(len(train_indices) * noise_ratio)
            noisy_indices = np.random.choice(len(train_indices), num_noisy, replace=False)
            for idx in noisy_indices:
                current_target = train_targets[idx]
                possible_targets = list(range(classes_per_client))
                possible_targets.remove(current_target)
                train_targets[idx] = np.random.choice(possible_targets)

            logging.info(f"Client {client_idx}: Added noise to {num_noisy} samples")

        # 创建训练集
        mapped_train_dataset = MappedDataset(
            dataset=train_dataset,
            indices=train_indices,
            new_targets=np.array(train_targets),
            class_mapping=class_mapping,
            original_targets=np.array(original_train_targets)
        )

        # 创建验证集
        mapped_val_dataset = MappedDataset(
            dataset=train_dataset,
            indices=val_indices,
            new_targets=np.array(val_targets),
            class_mapping=class_mapping,
            original_targets=np.array(original_val_targets)
        )

        # 验证数据集大小
        assert len(train_indices) == classes_per_client * 5, \
            f"Client {client_idx}: Expected {classes_per_client * 5} train samples, got {len(train_indices)}"
        assert len(val_indices) == classes_per_client, \
            f"Client {client_idx}: Expected {classes_per_client} val samples, got {len(val_indices)}"

        # 创建数据加载器
        train_loader = DataLoader(
            mapped_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            mapped_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 记录客户端数据统计
        logging.info(f"\nClient {client_idx} Data Statistics:")
        logging.info(f"  Is Noisy Client: {is_noisy}")
        logging.info(f"  Number of Classes: {len(set(train_targets))}")
        logging.info(f"  Training Samples: {len(train_indices)}")
        logging.info(f"  Validation Samples: {len(val_indices)}")
        if is_noisy:
            logging.info(f"  Noisy Samples: {num_noisy}")
            logging.info(f"  Clean Samples: {len(train_indices) - num_noisy}")
            logging.info(f"  Noise Ratio: {noise_ratio}")

        client_loaders.append((train_loader, val_loader))
        current_class_idx += classes_per_client

    return client_loaders


def setup_logging(log_dir: str, experiment_name: str):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    # 为噪声客户端0创建专门的日志文件
    noisy_client0_log = os.path.join(log_dir, "噪声客户端0.txt")


    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 噪声客户端0的日志处理器
    noisy_client0_handler = logging.FileHandler(noisy_client0_log)
    noisy_client0_handler.setFormatter(formatter)
    
    # 创建特殊的过滤器只记录客户端0的日志
    class Client0Filter(logging.Filter):
        def filter(self, record):
            return (
                "Client 0" in record.getMessage() or 
                "客户端 0" in record.getMessage() or
                "联邦学习第" in record.getMessage()
            )
    
    noisy_client0_handler.addFilter(Client0Filter())
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有的处理器
    logger.handlers.clear()

    # 添加新的处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(noisy_client0_handler)
    logging.info(f"Started logging to {log_file}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Noise client 0 logs will be saved to: {noisy_client0_log}")

def save_federated_model(
        model,
        save_dir: str,
        round: int,
        metrics: Dict,
        is_final: bool = False
):
    """
    保存联邦学习模型

    Args:
        model: 模型
        save_dir: 保存目录
        round: 当前轮次
        metrics: 训练指标
        is_final: 是否为最终模型
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建文件名
    if is_final:
        filename = "final_federated_model.pth"
    else:
        filename = f"federated_model_round_{round}.pth"

    save_path = os.path.join(save_dir, filename)

    # 保存模型
    state = {
        'round': round,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(state, save_path)

    if is_final:
        logging.info(f"\nSaved final model to {save_path}")
    else:
        logging.info(f"\nSaved round {round} model to {save_path}")

    # 记录模型性能
    logging.info("Model Performance:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.4f}")
            else:
                logging.info(f"  {key}: {value}")