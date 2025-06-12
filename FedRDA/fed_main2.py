# fed_main2.py

import os
import torch
import argparse
import logging
import random
import numpy as np
import pandas as pd  # 新增: 导入pandas处理Excel
import sys  # 新增: 导入sys用于退出程序
from datetime import datetime
from configs.config import get_args
from utils.dataset import FingerVeinDataset, create_client_data_loaders
from utils.client_detector import NoiseClientDetector
from models.resnet import create_model
from federated.client import FederatedClient
from federated.server import FederatedServer
from federated.utils import setup_logging, save_federated_model
import torchvision.transforms as transforms
from display_results import visualize_results
import time

def get_fed_args():
    """获取联邦学习相关参数"""
    parser = argparse.ArgumentParser(description='Federated Learning for Finger Vein Classification')
    
    # 添加数据集选择参数
    parser.add_argument('--dataset_name', type=str, default='all',
                       choices=['all', 'SDUMLA', 'FV-USM', 'HKPU', 'MMCBNU', 'NUPT-FV'],
                       help='选择特定数据集或全部 (all, SDUMLA, FV-USM, HKPU, MMCBNU, NUPT-FV)')
    
    args = get_args()
    
    # 更新命名空间
    fed_args = parser.parse_args()
    for arg in vars(args):
        if not hasattr(fed_args, arg):
            setattr(fed_args, arg, getattr(args, arg))

    return fed_args

def initialize_global_model(args, num_classes_per_client):
    """初始化全局模型

    不再依赖预训练结果,直接初始化一个新模型

    Args:
        args: 配置参数
        num_classes_per_client: 每个客户端的类别数
    Returns:
        nn.Module: 初始化的全局模型
    """
    try:
        # 创建新的全局模型
        global_model = create_model(args, num_classes=num_classes_per_client)
        global_model = global_model.to(args.device)

        # 保存初始全局模型
        initial_model_path = os.path.join(args.save_dir, 'initial_global_model.pth')
        torch.save(global_model.state_dict(), initial_model_path)
        logging.info(f"\n保存初始全局模型至: {initial_model_path}")

        return global_model

    except Exception as e:
        logging.error(f"初始化全局模型失败: {str(e)}")
        raise e

def export_independent_training_metrics_to_excel(independent_metrics, save_dir):
    """
    导出独立训练阶段的指标到Excel文件
    
    Args:
        independent_metrics: 独立训练阶段的指标数据
        save_dir: 保存目录
    """
    try:
        # 确保保存目录存在
        results_dir = os.path.join(save_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备数据
        all_metrics = []
        
        for client_id, rounds_metrics in independent_metrics.items():
            for round_data in rounds_metrics:
                metrics_row = {
                    'client_id': client_id,
                    'round': round_data['round'],
                    'train_loss': round_data['train_metrics']['train_loss'],
                    'train_accuracy': round_data['train_metrics']['train_accuracy'],
                    'val_loss': round_data['val_metrics']['val_loss'],
                    'val_accuracy': round_data['val_metrics']['val_accuracy'],
                    'is_noisy': round_data['train_metrics']['is_noisy']
                }
                all_metrics.append(metrics_row)
        
        # 创建DataFrame
        df = pd.DataFrame(all_metrics)
        
        # 保存到Excel
        excel_path = os.path.join(results_dir, f'independent_training_metrics_{timestamp}.xlsx')
        df.to_excel(excel_path, index=False)
        
        logging.info(f"\n成功导出第一阶段训练指标到: {excel_path}")
        return excel_path
        
    except Exception as e:
        logging.error(f"导出训练指标到Excel失败: {str(e)}")
        return None

def main():
    # 获取参数
    args = get_fed_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置实验名称和日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"federated_training_{args.dataset_name}_{timestamp}"
    setup_logging(args.log_dir, experiment_name)

    # 记录配置
    logging.info("启动联邦学习训练")
    logging.info(f"配置信息:")
    logging.info(f"选择的数据集: {args.dataset_name}")
    logging.info(f"Alpha: {args.alpha}")
    logging.info(f"Beta start: {args.beta_start}")
    logging.info(f"Beta end: {args.beta_end}")
    logging.info(f"Detection start round: {args.detection_start_round}")
    logging.info(f"Trust threshold: {args.trust_threshold}")
    logging.info(f"Number of noisy clients: {args.num_noisy_clients}")
    logging.info(f"Noise ratio: {args.noise_ratio}")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    try:
        # 创建数据集
        logging.info("正在加载数据集...")
        train_dataset = FingerVeinDataset(
            root_dir=args.data_root,
            dataset_name=args.dataset_name,  # 使用选择的数据集
            transform=transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        # 验证数据集信息
        logging.info(f"成功加载数据集: {train_dataset.num_classes}类, 每类6个样本")
        
        # 确定噪声客户端的索引
        noisy_client_indices = random.sample(range(args.num_clients), min(args.num_noisy_clients, args.num_clients))
        logging.info(f"噪声客户端索引: {noisy_client_indices}")

        # 创建客户端数据加载器
        logging.info("正在创建客户端数据加载器...")
        client_loaders = create_client_data_loaders(
            train_dataset=train_dataset,
            val_dataset=None,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            noise_ratio=args.noise_ratio,
            noisy_client_ids=noisy_client_indices,
            seed=args.seed
        )

        # 计算每个客户端的平均类别数
        total_classes = train_dataset.num_classes
        base_classes_per_client = total_classes // args.num_clients
        remaining_classes = total_classes % args.num_clients
        
        if remaining_classes > 0:
            logging.info(f"每个客户端基础类别数: {base_classes_per_client}，前 {remaining_classes} 个客户端将多分配一个类别")
        else:
            logging.info(f"每个客户端均分配 {base_classes_per_client} 个类别")

        # 初始化全局模型（使用基础类别数，服务器只需要知道每个客户端的基本类别数）
        logging.info("\n初始化全局模型...")
        global_model = initialize_global_model(args, base_classes_per_client)

        # 创建服务器
        logging.info("正在初始化联邦学习服务器...")
        server = FederatedServer(
            model=global_model,
            num_clients=args.num_clients,
            device=args.device,
            args=args,
            save_dir=args.save_dir
        )

        # 创建客户端
        logging.info("正在初始化客户端...")
        clients = []
        for i in range(args.num_clients):
            train_loader, val_loader = client_loaders[i]
            
            # 获取当前客户端的类别数
            client_classes = base_classes_per_client + (1 if i < remaining_classes else 0)
            logging.info(f"客户端 {i} 的类别数: {client_classes}")

            # 创建客户端模型（使用实际分配的类别数）
            client_model = create_model(args, num_classes=client_classes)
            client_model = client_model.to(args.device)

            client = FederatedClient(
                args=args,
                client_id=i,
                model=client_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=args.device,
                is_noisy=i in noisy_client_indices,
                noise_ratio=args.noise_ratio if i in noisy_client_indices else 0.0,
                learning_rate=args.fed_lr,
                local_epochs=args.local_epochs,
                loss_type=args.loss_type,
                num_classes=client_classes
            )
            clients.append(client)
            logging.info(f"初始化客户端 {i} 完成")

        # 联邦学习主循环
        logging.info("\n开始联邦学习训练...")
        total_start_time = time.time()

        # 第一阶段: 客户端独立训练 Tw 轮
        logging.info(f"\n第一阶段：客户端独立训练 {args.Tw} 轮...")
        independent_metrics = {i: [] for i in range(args.num_clients)}

        # 收集所有轮次的指标
        all_train_metrics = []  # 所有轮次的训练指标
        all_val_metrics = []  # 所有轮次的验证指标

        for client_idx, client in enumerate(clients):
            client_start_time = time.time()
            logging.info(f"\n客户端 {client_idx} 开始独立训练...")

            # 客户端独立训练Tw轮
            for local_round in range(1, args.Tw + 1):
                train_result = client.local_train(
                    args=args,
                    independent_round=local_round
                )
                val_result = client.evaluate()

                # 记录训练指标
                independent_metrics[client_idx].append({
                    'round': local_round,
                    'train_metrics': train_result,
                    'val_metrics': val_result
                })

                # 收集每一轮的指标用于更新服务器
                all_train_metrics.append(train_result)
                all_val_metrics.append(val_result)

                # 更新服务器的指标
                server.process_local_training_results(all_train_metrics, all_val_metrics)

                logging.info(f"轮次 {local_round}/{args.Tw}:")
                logging.info(f"训练损失: {train_result['train_loss']:.4f}")
                logging.info(f"训练准确率: {train_result['train_accuracy']:.2f}%")
                logging.info(f"验证准确率: {val_result['val_accuracy']:.2f}%")

            client_time = time.time() - client_start_time
            logging.info(f"客户端 {client_idx} 独立训练完成，用时: {client_time:.2f}s")

        # 第一阶段结束，导出指标到Excel文件
        excel_path = export_independent_training_metrics_to_excel(independent_metrics, args.save_dir)
        if excel_path:
            logging.info(f"第一阶段训练指标已成功导出到Excel文件: {excel_path}")
        
        # 第一阶段结束后停止程序
        logging.info("第一阶段训练已完成，程序将按需求终止")
        sys.exit(0)

        # 以下代码不再执行
        # 第二阶段: 噪声客户端检测
        logging.info("\n第二阶段：服务端进行噪声客户端检测...")
        detected_noisy_clients = server.detect_noisy_clients()  # 这里 server 会基于之前累积的指标进行检测
        logging.info(f"检测到的噪声客户端: {detected_noisy_clients}")

        # 第三阶段
        logging.info(f"\n第三阶段：开始联邦学习训练 {args.fed_rounds} 轮...")
        server.current_round = 0  # 重置轮次计数器
        
        for fed_round in range(1, args.fed_rounds + 1):
            round_start_time = time.time()
            logging.info(f"\n联邦学习轮次 {fed_round}/{args.fed_rounds}")
            server.current_round = fed_round  # 更新轮次

            # 分发全局模型参数
            global_parameters = server.get_parameters()
            for client in clients:
                client.set_parameters(global_parameters)

            # 客户端本地训练
            train_metrics = []
            val_metrics = []
            client_parameters = []

            for client_idx, client in enumerate(clients):
                is_noisy = client_idx in detected_noisy_clients
                logging.info(f"Client {client_idx} - {'Noisy' if is_noisy else 'Clean'}")

                # 执行local_epochs次本地训练并进行噪声样本检测
                train_result = client.federated_train(
                    args=args,
                    fed_round=fed_round,
                    is_detected_noisy=is_noisy,
                    tw_metrics=independent_metrics[client_idx]  # 传入前Tw轮的训练指标
                )
                val_result = client.evaluate()

                train_metrics.append(train_result)
                val_metrics.append(val_result)
                client_parameters.append((client.get_parameters(), train_result['num_samples']))

            # 聚合模型参数，这里要考虑噪声客户端的权重
            aggregated_parameters = server.aggregate_parameters(
                client_parameters,
                detected_noisy_clients  # 传入检测到的噪声客户端集合
            )
            server.update_global_model(aggregated_parameters)

            # 更新和记录指标
            server.update_metrics(train_metrics, val_metrics)

            round_time = time.time() - round_start_time
            logging.info(f"联邦学习轮次 {fed_round} 完成，用时: {round_time:.2f}s")

            # 保存检查点
            if fed_round % args.fed_save_freq == 0:
                save_federated_model(
                    server.model,
                    args.save_dir,
                    fed_round,
                    server.train_metrics_history[-1]
                )

        total_time = time.time() - total_start_time
        logging.info(f"\n训练完成，总用时: {total_time:.2f}s")

        # 保存最终结果
        save_federated_model(
            server.model,
            args.save_dir,
            args.fed_rounds,
            server.train_metrics_history[-1],
            is_final=True
        )
        logging.info(f"\n保存最终模型")

#         # 生成可视化结果
#         try:
#             results_dir = os.path.join(args.save_dir, 'results')
#             os.makedirs(results_dir, exist_ok=True)
#             visualize_results(server.train_metrics_history, results_dir)
#             logging.info(f"\n可视化结果已保存到: {results_dir}")

#         except Exception as viz_error:
#             logging.error(f"生成可视化结果时出错: {str(viz_error)}")
#             logging.error("继续执行其他操作...")

    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()