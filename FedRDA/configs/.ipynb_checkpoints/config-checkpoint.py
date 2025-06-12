# configs/config.py

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning with Noise Detection')

    # 数据集参数
    parser.add_argument('--data_root', default='./data/data_Preprocessed', help='数据集根目录')
    parser.add_argument('--num_classes', type=int, default=2880, help='总类别数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--batch_size', type=int, default=128, help='训练批次大小')
    parser.add_argument('--img_dim', type=int, default=224, help='输入图像的尺寸')

    # 联邦学习参数
    parser.add_argument('--num_clients', type=int, default=5, help='客户端数量')
    parser.add_argument('--fed_rounds', type=int, default=50, help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--fed_lr', type=float, default=0.001, help='联邦学习学习率')
    parser.add_argument('--fed_momentum', type=float, default=0.9, help='联邦学习动量系数')
    parser.add_argument('--fed_weight_decay', type=float, default=5e-4, help='联邦学习权重衰减')
    parser.add_argument('--fed_save_freq', type=int, default=30, help='联邦学习模型保存频率')

    # 噪声检测相关参数
    parser.add_argument('--num_noisy_clients', type=int, default=2, help='噪声客户端数量')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='噪声比例')
    parser.add_argument('--detection_start_round', type=int, default=6, help='开始噪声检测的轮次')
    parser.add_argument('--detection_window_size', type=int, default=5, help='损失曲线窗口大小')
    parser.add_argument('--trust_threshold', type=float, default=0.6, help='检测置信度阈值')

    # RDA参数
    parser.add_argument('--alpha', type=float, default=0.1, help='标签松弛参数')
    parser.add_argument('--beta_start', type=float, default=0.5, help='初始置信度阈值')
    parser.add_argument('--beta_end', type=float, default=0.8, help='最终置信度阈值')

    # 不确定性训练参数 (与RDA配合使用)
    parser.add_argument('--gmm_tau', type=float, default=0.5, help='GMM聚类阈值')
    parser.add_argument('--min_samples', type=int, default=10, help='GMM拟合最小样本数')

    # 设备参数
    parser.add_argument('--device', default='cuda', help='设备：cuda或cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 保存和日志参数
    parser.add_argument('--save_dir', default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', default='./logs', help='日志保存目录')

    # Loss类型
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'gce', 'rda'],
                        help='损失函数类型: ce/gce/rda')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='ResNet架构类型')
    # 解析参数
    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args