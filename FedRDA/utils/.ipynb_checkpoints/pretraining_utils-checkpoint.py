# utils/pretraining_utils.py

import torch
from torch.utils.data import DataLoader
import logging
from typing import Tuple, Dict
from .simclr_pretraining import (
    ContrastiveDataset,
    SimCLR,
    train_simclr,
    extract_features,
    compute_knn_indices
)
import numpy as np


def setup_simclr_pretraining(
        args,
        train_dataset,
        model_creator
) -> Tuple[Dict, torch.Tensor, np.ndarray]:
    """设置并运行SimCLR预训练

    Args:
        args: 配置参数
        train_dataset: 训练数据集
        model_creator: 模型创建函数

    Returns:
        Dict: 包含预训练模型的结果字典
            {
                'model': 预训练模型,
                'features': 特征表示,
                'knn_indices': KNN索引,
                'pretrain_path': 预训练模型保存路径,
                'feature_path': 特征保存路径
            }
    """
    # 记录预训练配置信息
    logging.info(f"\nStarting Client Pre-training...")
    logging.info(f"Dataset size: {len(train_dataset)}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Pre-train epochs: {args.pretrain_epochs}")
    logging.info(f"Learning rate: {args.simclr_lr}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"KNN neighbors: {args.k_neighbors}")

    # 包装数据集
    contrast_dataset = ContrastiveDataset(train_dataset)

    # SimCLR预训练数据加载器
    train_loader = DataLoader(
        contrast_dataset,
        batch_size=args.simclr_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 创建模型
    backbone = model_creator(args)
    # SimCLR模型配置
    simclr_model = SimCLR(
        backbone=backbone,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim
    )
    # 计算并记录模型参数量
    total_params = sum(p.numel() for p in simclr_model.parameters())
    trainable_params = sum(p.numel() for p in simclr_model.parameters() if p.requires_grad)
    logging.info(f"Model architecture:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")

    # 训练模型
    trained_model = train_simclr(
        args,
        model=simclr_model,
        train_loader=train_loader,
        epochs=args.pretrain_epochs,
        device=args.device,
        learning_rate=args.simclr_lr,
        temperature=args.temperature,
        weight_decay=args.simclr_weight_decay,
        logging_interval=5,
        k=args.k_neighbors
    )

    # 提取特征
    logging.info("\nExtracting features...")
    features = extract_features(trained_model, train_loader, args.device)
    logging.info(f"Features shape: {features.shape}")

    # 计算KNN索引
    logging.info("\nComputing KNN indices...")
    knn_indices = compute_knn_indices(
        features,
        k=args.k_neighbors  # 使用args中的参数
    )
    logging.info(f"KNN indices shape: {knn_indices.shape}")

    # 保存预训练结果
    pretrain_path = f"{args.pretrain_save_dir}/pretrain_model.pth"
    feature_path = f"{args.feature_save_dir}/features.pth"

    torch.save(trained_model.state_dict(), pretrain_path)
    torch.save({
        'features': features,
        'knn_indices': knn_indices
    }, feature_path)

    logging.info(f"\nSaved pre-trained model to: {pretrain_path}")
    logging.info(f"Saved features to: {feature_path}")

    # 返回预训练结果字典
    pretrain_results = {
        'model': trained_model,
        'features': features,
        'knn_indices': knn_indices,
        'pretrain_path': pretrain_path,
        'feature_path': feature_path
    }

    return pretrain_results