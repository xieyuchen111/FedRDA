# federated/server.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
import logging
from collections import OrderedDict
import copy
import os
from datetime import datetime
from utils.client_detector import NoiseClientDetector
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

class FederatedServer:
    def __init__(
            self,
            model: nn.Module,
            num_clients: int,
            device: str,
            args,
            save_dir: str = './checkpoints'
    ):
        """
        联邦学习服务器
        Args:
            model: 全局模型
            num_clients: 客户端总数
            device: 设备
            args: 配置参数
            detection_begin_epoch: 开始噪声检测的轮次
            save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.num_clients = num_clients
        self.device = device
        self.save_dir = save_dir
        self.current_round = 0

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 记录训练指标
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.client_performance_history = {}
        self.detection_start_round = args.detection_start_round
        # 初始化噪声检测器
        self.noise_detector = NoiseClientDetector(
            detection_start_round=args.detection_start_round,
            window_size=args.detection_window_size
        )

        # 初始化日志
        self._initialize_logging()

    def _initialize_logging(self):
        """初始化服务器日志"""
        logging.info(f"\nInitializing Federated Learning Server:")
        logging.info(f"Number of Clients: {self.num_clients}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Noise Client Detection Start Round: {self.detection_start_round}")
        logging.info(f"Model Save Directory: {self.save_dir}")

    def aggregate_parameters(
            self,
            client_parameters: List[Tuple[Dict[str, torch.Tensor], int]],
            round_num: int
    ) -> Dict[str, torch.Tensor]:
        """
        聚合客户端参数
        Args:
            client_parameters: 客户端参数列表，每个元素为(参数字典, 样本数量)
            round_num: 当前轮次
        Returns:
            Dict: 聚合后的参数字典
        """
        logging.info(f"\nAggregating parameters for round {round_num}")

        # 计算总样本数
        total_samples = sum(samples for _, samples in client_parameters)
        aggregated_params = OrderedDict()

        try:
            # 初始化聚合参数字典
            for name, param in client_parameters[0][0].items():
                aggregated_params[name] = torch.zeros_like(
                    param,
                    dtype=param.dtype,
                    device=self.device
                )

            # 记录每个客户端的贡献
            client_contributions = []

            # 获取当前检测到的噪声客户端
            if round_num >= self.detection_start_round:  # 修改这里
                noisy_clients = self.noise_detector.detect_noisy_clients(round_num)
            else:
                noisy_clients = set()

            # 加权平均聚合，降低噪声客户端的权重
            for client_idx, (parameters, num_samples) in enumerate(client_parameters):
                # 如果是噪声客户端，降低其权重
                if client_idx in noisy_clients:
                    weight = (num_samples / total_samples) * 0.5  # 降低权重为原来的一半
                else:
                    weight = num_samples / total_samples

                client_contributions.append((client_idx, weight))

                for name, param in parameters.items():
                    if param.dtype == torch.long:
                        weighted_param = (param.float() * weight).long()
                    else:
                        weighted_param = param.to(self.device) * weight

                    if aggregated_params[name].dtype == torch.long:
                        aggregated_params[name] += weighted_param.long()
                    else:
                        aggregated_params[name] += weighted_param

            # 记录聚合信息
            logging.info("Parameter aggregation weights:")
            for client_idx, weight in client_contributions:
                client_type = "Noisy" if client_idx in noisy_clients else "Clean"
                logging.info(f"Client {client_idx} ({client_type}): {weight:.4f}")

            return aggregated_params

        except Exception as e:
            logging.error(f"Error during parameter aggregation: {str(e)}")
            raise e

    def update_global_model(self, aggregated_parameters: Dict[str, torch.Tensor]):
        """
        更新全局模型参数
        Args:
            aggregated_parameters: 聚合后的参数
        """
        try:
            self.model.load_state_dict(aggregated_parameters)
            self.current_round += 1
            logging.info(f"Global model updated to round {self.current_round}")
        except Exception as e:
            logging.error(f"Error updating global model: {str(e)}")
            raise e

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数"""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def update_metrics(self, train_metrics: List[Dict], val_metrics: List[Dict]):
        """
        更新训练和验证指标
        Args:
            train_metrics: 训练指标列表
            val_metrics: 验证指标列表
        """
        logging.info(f"\nRound {self.current_round} Results Summary:")
        logging.info("=" * 70)

        # 1. 客户端详细性能
        logging.info("\nPer-Client Performance:")
        clean_clients_metrics = []
        noisy_clients_metrics = []

        # 更新噪声检测器的指标
        client_round_metrics = {}

        for client_id in range(self.num_clients):
            train_metric = next(m for m in train_metrics if m['client_id'] == client_id)
            val_metric = next(m for m in val_metrics if m['client_id'] == client_id)

            is_noisy = train_metric.get('is_noisy_client', False)
            metrics_list = noisy_clients_metrics if is_noisy else clean_clients_metrics
            client_metrics = {
                'client_id': client_id,
                'train_loss': train_metric['train_loss'],
                'train_accuracy': train_metric.get('train_accuracy', 0.0),  # 添加默认值
                'val_loss': val_metric['val_loss'],
                'val_accuracy': val_metric['val_accuracy'],
            }
            metrics_list.append(client_metrics)

            # 更新噪声检测器指标
            client_round_metrics[client_id] = {
                'val_accuracy': val_metric['val_accuracy'],
                'val_loss': val_metric['val_loss'],
                'train_loss': train_metric['train_loss'],
                'train_accuracy': train_metric.get('train_accuracy', 0.0)
            }

            # 记录客户端性能
            logging.info(f"\nClient {client_id} ({'Noisy' if is_noisy else 'Clean'}):")
            logging.info(f"  Training:")
            logging.info(f"    Loss: {train_metric['train_loss']:.4f}")
            logging.info(f"    Accuracy: {train_metric.get('train_accuracy', 0.0):.2f}%")
            # # # 修改这里的检测判断
            # if self.current_round >= self.detection_start_round + 1:
            #     logging.info(f"    Pseudo Labels: {train_metric.get('num_pseudo_labels', 0)}")
            #     logging.info(f"    Detected Noisy: {train_metric.get('num_noisy_detected', 0)}")

            logging.info(f"  Validation:")
            logging.info(f"    Loss: {val_metric['val_loss']:.4f}")
            logging.info(f"    Accuracy: {val_metric['val_accuracy']:.2f}%")

        # 更新噪声检测器
        self.noise_detector.update_metrics(self.current_round, client_round_metrics)

        # 执行噪声检测
        if self.current_round >= self.detection_start_round:
            detected_noisy_clients = self.noise_detector.detect_noisy_clients(self.current_round)
            if detected_noisy_clients:
                logging.info(f"\nDetected noisy clients in round {self.current_round}:")
                logging.info(f"Noisy client IDs: {sorted(detected_noisy_clients)}")

                # 获取详细统计信息
                detection_stats = self.noise_detector.get_detection_stats()
                logging.info("\nNoise Detection Statistics:")
                logging.info(f"Total clients: {detection_stats['total_clients']}")
                logging.info(f"Number of noisy clients: {detection_stats['noisy_clients']}")
                if 'clean_clients' in detection_stats:
                    logging.info("\nClean Client Metrics:")
                    logging.info(f"Average training loss: "
                                 f"{detection_stats['clean_clients']['avg_train_loss']:.4f}")
                    logging.info(f"Training loss std: "
                                 f"{detection_stats['clean_clients']['std_train_loss']:.4f}")
                if 'noisy_clients' in detection_stats:
                    logging.info("\nNoisy Client Metrics:")
                    logging.info(f"Average training loss: "
                                 f"{detection_stats['noisy_clients']['avg_train_loss']:.4f}")
                    logging.info(f"Training loss std: "
                                 f"{detection_stats['noisy_clients']['std_train_loss']:.4f}")
                if 'clustering' in detection_stats:
                    logging.info("\nClustering Information:")
                    logging.info(f"Cluster sizes: {detection_stats['clustering']['cluster_sizes']}")

        # 2. 计算并记录整体统计信息
        overall_stats = self._calculate_overall_stats(
            clean_clients_metrics,
            noisy_clients_metrics
        )

        # 3. 保存历史记录
        self.train_metrics_history.append({
            'round': self.current_round,
            'overall_stats': overall_stats,
            'clean_clients': clean_clients_metrics,
            'noisy_clients': noisy_clients_metrics
        })

        # 4. 保存检查点
        self._save_checkpoint(overall_stats)

    def _calculate_overall_stats(self, clean_metrics: List[Dict], noisy_metrics: List[Dict]) -> Dict:
        """计算整体统计信息"""
        stats = {
            'all_clients': {
                'train_loss': np.mean([m['train_loss'] for m in clean_metrics + noisy_metrics]),
                'train_accuracy': np.mean([m['train_accuracy'] for m in clean_metrics + noisy_metrics]),
                'val_loss': np.mean([m['val_loss'] for m in clean_metrics + noisy_metrics]),
                'val_accuracy': np.mean([m['val_accuracy'] for m in clean_metrics + noisy_metrics])
            }
        }

        if clean_metrics:
            stats['clean_clients'] = {
                'train_loss': np.mean([m['train_loss'] for m in clean_metrics]),
                'train_accuracy': np.mean([m['train_accuracy'] for m in clean_metrics]),
                'val_loss': np.mean([m['val_loss'] for m in clean_metrics]),
                'val_accuracy': np.mean([m['val_accuracy'] for m in clean_metrics])
            }

        if noisy_metrics:
            stats['noisy_clients'] = {
                'train_loss': np.mean([m['train_loss'] for m in noisy_metrics]),
                'train_accuracy': np.mean([m['train_accuracy'] for m in noisy_metrics]),
                'val_loss': np.mean([m['val_loss'] for m in noisy_metrics]),
                'val_accuracy': np.mean([m['val_accuracy'] for m in noisy_metrics])
            }

        self._log_overall_stats(stats)
        return stats

    def _log_overall_stats(self, stats: Dict):
        """记录整体统计信息"""
        logging.info("\nOverall Statistics:")
        for client_type, metrics in stats.items():
            logging.info(f"\n{client_type.replace('_', ' ').title()}:")
            logging.info(f"  Training Loss: {metrics['train_loss']:.4f}")
            logging.info(f"  Training Accuracy: {metrics['train_accuracy']:.2f}%")
            logging.info(f"  Validation Loss: {metrics['val_loss']:.4f}")
            logging.info(f"  Validation Accuracy: {metrics['val_accuracy']:.2f}%")

    def _save_checkpoint(self, stats: Dict):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.save_dir,
            f"federated_checkpoint_round_{self.current_round}.pth"
        )

        torch.save({
            'round': self.current_round,
            'model_state_dict': self.model.state_dict(),
            'metrics': stats,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, checkpoint_path)

        logging.info(f"\nSaved checkpoint to {checkpoint_path}")

    def get_best_metrics(self) -> Dict:
        """获取最佳指标"""
        if not self.train_metrics_history:
            return {}

        # 根据验证准确率选择最佳轮次
        best_round_idx = np.argmax([
            m['overall_stats']['all_clients']['val_accuracy']
            for m in self.train_metrics_history
        ])
        best_metrics = self.train_metrics_history[best_round_idx]

        return {
            'best_round': best_metrics['round'],
            'best_metrics': best_metrics['overall_stats'],
            'clean_clients': best_metrics['clean_clients'],
            'noisy_clients': best_metrics['noisy_clients']
        }

    def get_training_progress(self) -> Dict:
        """获取训练进度信息"""
        if not self.train_metrics_history:
            return {}

        return {
            'current_round': self.current_round,
            'total_rounds': len(self.train_metrics_history),
            'best_metrics': self.get_best_metrics(),
            'latest_metrics': self.train_metrics_history[-1]['overall_stats'],
            'noisy_clients': list(
                self.noise_detector.noisy_clients if self.current_round >= self.detection_start_round else set())
        }

    def get_detection_summary(self) -> Dict:
        """获取噪声检测总结信息"""
        if self.current_round < self.detection_start_round:
            return {'message': 'Noise detection has not started yet.'}

        stats = self.noise_detector.get_detection_stats()
        detected_noisy = list(self.noise_detector.noisy_clients)

        summary = {
            'total_clients': stats['total_clients'],
            'num_noisy_detected': len(detected_noisy),
            'noisy_client_ids': detected_noisy,
            'detection_started_round': self.detection_start_round,
            'current_round': self.current_round,
            # 'global_metrics': stats['global_val_accuracy']
        }

        if 'clean_clients' in stats:
            summary['clean_clients_metrics'] = stats['clean_clients']
        if 'noisy_clients' in stats:
            summary['noisy_clients_metrics'] = stats['noisy_clients']

        return summary
    def save_metrics_to_excel(self, save_path=None):
        """
        将所有客户端每轮的训练指标保存到Excel文件

        Args:
            save_path: Excel文件保存路径，默认为None，将使用save_dir
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'federated_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

        # 创建一个Excel工作簿
        wb = Workbook()

        # 创建汇总表
        summary_sheet = wb.active
        summary_sheet.title = "Summary"

        # 添加汇总标题
        summary_sheet.append(["Round", "Overall Train Loss", "Overall Train Accuracy", 
                               "Overall Val Loss", "Overall Val Accuracy",
                               "Clean Clients Train Loss", "Clean Clients Train Accuracy",
                               "Clean Clients Val Loss", "Clean Clients Val Accuracy",
                               "Noisy Clients Train Loss", "Noisy Clients Train Accuracy",
                               "Noisy Clients Val Loss", "Noisy Clients Val Accuracy"])

        # 填充汇总数据
        for metrics in self.train_metrics_history:
            round_num = metrics['round']
            overall = metrics['overall_stats']['all_clients']

            row_data = [round_num, 
                        overall['train_loss'], overall['train_accuracy'],
                        overall['val_loss'], overall['val_accuracy']]

            # 添加干净客户端数据
            if 'clean_clients' in metrics['overall_stats']:
                clean = metrics['overall_stats']['clean_clients']
                row_data.extend([clean['train_loss'], clean['train_accuracy'],
                                 clean['val_loss'], clean['val_accuracy']])
            else:
                row_data.extend([None, None, None, None])

            # 添加噪声客户端数据
            if 'noisy_clients' in metrics['overall_stats']:
                noisy = metrics['overall_stats']['noisy_clients']
                row_data.extend([noisy['train_loss'], noisy['train_accuracy'],
                                 noisy['val_loss'], noisy['val_accuracy']])
            else:
                row_data.extend([None, None, None, None])

            summary_sheet.append(row_data)

        # 为每个客户端创建单独的表格
        client_metrics = {}
        for round_metrics in self.train_metrics_history:
            round_num = round_metrics['round']

            # 收集所有客户端数据
            for client_type in ['clean_clients', 'noisy_clients']:
                if client_type in round_metrics:
                    for client in round_metrics[client_type]:
                        client_id = client['client_id']
                        if client_id not in client_metrics:
                            client_metrics[client_id] = []

                        client_metrics[client_id].append({
                            'round': round_num,
                            'is_noisy': client_type == 'noisy_clients',
                            'train_loss': client['train_loss'],
                            'train_accuracy': client['train_accuracy'],
                            'val_loss': client['val_loss'],
                            'val_accuracy': client['val_accuracy']
                        })

        # 为每个客户端创建工作表
        for client_id, metrics in client_metrics.items():
            # 创建DataFrame
            df = pd.DataFrame(metrics)

            # 创建新工作表
            sheet = wb.create_sheet(title=f"Client_{client_id}")

            # 写入数据
            for r in dataframe_to_rows(df, index=False, header=True):
                sheet.append(r)

        # 保存Excel文件
        wb.save(save_path)
        logging.info(f"\nSaved client metrics to Excel: {save_path}")

        return save_path