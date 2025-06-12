# utils/client_detector.py
import torch
import numpy as np
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class NoiseClientDetector:
    def __init__(
            self,
            detection_start_round: int = 5,
            window_size: int = 5,
            min_samples_for_clustering: int = 3,
            save_dir: str = './detection_results',
            beta=0,
            trust_threshold=0
    ):
        self.detection_start_round = detection_start_round
        self.window_size = window_size
        self.min_samples_for_clustering = min_samples_for_clustering
        self.save_dir = save_dir
        self.detection_completed = False  # 新添加的标志变量
        # 存储历史指标
        self.client_history = defaultdict(list)
        # 存储检测结果
        self.noisy_clients = set()
        # 存储聚类结果历史
        self.clustering_history = []

        # 初始化全局统计信息
        self.global_stats = {
            'avg_val_accuracy': [],
            'avg_train_loss': [],
            # 'avg_credal_ratio': [],
            # 'avg_prediction_consistency': []
        }

        # 初始化日志
        self._setup_logger()

    def _setup_logger(self):
        """设置日志"""
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger("NoiseClientDetector")

    def update_metrics(self, round_num: int, client_metrics: Dict) -> None:
        """
        更新客户端指标

        Args:
            round_num: 当前轮次
            client_metrics: 客户端指标字典
        """
        try:
            # 收集所有客户端的指标
            val_accuracies = []
            train_losses = []
            credal_ratios = []
            prediction_consistencies = []

            for client_id, metrics in client_metrics.items():
                # 保存客户端历史指标
                self.client_history[client_id].append({
                    'round': round_num,
                    'train_loss': metrics['train_loss'],
                    'val_accuracy': metrics['val_accuracy'],
                    # 'credal_metrics': metrics['credal_metrics']
                })

                # 收集指标用于计算全局平均值
                val_accuracies.append(metrics['val_accuracy'])
                train_losses.append(metrics['train_loss'])
                # credal_ratios.append(metrics['credal_metrics']['credal_ratio'])
                # prediction_consistencies.append(metrics['credal_metrics']['prediction_consistency'])

            # 更新全局统计信息
            self.global_stats['avg_val_accuracy'].append(np.mean(val_accuracies))
            self.global_stats['avg_train_loss'].append(np.mean(train_losses))
            # self.global_stats['avg_credal_ratio'].append(np.mean(credal_ratios))
            # self.global_stats['avg_prediction_consistency'].append(np.mean(prediction_consistencies))

            self.logger.info(f"Round {round_num}: Updated metrics for {len(client_metrics)} clients")

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            raise e

    def get_client_features(self) -> Tuple[np.ndarray, List[int]]:
        """
        提取客户端特征向量用于聚类，只使用训练损失相关特征

        Returns:
            Tuple[np.ndarray, List[int]]: 特征矩阵和对应的客户端ID列表
        """
        features = []
        client_ids = []

        for client_id, history in self.client_history.items():
            if len(history) >= self.window_size:
                recent_history = history[-self.window_size:]

                # 只使用损失相关的特征
                avg_train_loss = np.mean([m['train_loss'] for m in recent_history])
                loss_std = np.std([m['train_loss'] for m in recent_history])
                loss_trend = recent_history[-1]['train_loss'] - recent_history[0]['train_loss']

                # 构建特征向量（只包含损失相关特征）
                feature = np.array([
                    avg_train_loss,  # 平均训练损失
                    # loss_std,  # 损失标准差
                    # loss_trend,  # 损失变化趋势
                ])

                features.append(feature)
                client_ids.append(client_id)

        return np.array(features), client_ids

    def detect_noisy_clients(self, round_num: int) -> Set[int]:
        """
        使用聚类方法检测噪声客户端
        Args:
            round_num: 当前轮次
        Returns:
            Set[int]: 检测到的噪声客户端ID集合
        """
        # 如果已经完成检测，直接返回之前的结果
        if self.detection_completed:
            return self.noisy_clients
            
        # 如果未到检测轮次，返回空集合
        if round_num != self.detection_start_round:
            return set()
    
        try:
            # 获取特征向量
            features, client_ids = self.get_client_features()
    
            # 检查数据是否足够
            if len(features) < self.min_samples_for_clustering:
                return set()
    
            # 特征标准化
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
    
            # K-means聚类
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normalized_features)
            cluster_centers = kmeans.cluster_centers_
    
            # 确定噪声簇
            noisy_cluster = 0 if cluster_centers[0][0] > cluster_centers[1][0] else 1
    
            # 更新检测结果
            self.noisy_clients = {
                client_ids[i]
                for i in range(len(client_ids))
                if labels[i] == noisy_cluster
            }
    
            # 标记检测已完成
            self.detection_completed = True
    
            return self.noisy_clients
    
        except Exception as e:
            self.logger.error(f"Error in noise detection: {str(e)}")
            return set()

    def _log_clustering_results(
            self,
            features: np.ndarray,
            normalized_features: np.ndarray,
            labels: np.ndarray,
            client_ids: List[int],
            noisy_cluster: int,
            cluster_centers: np.ndarray
    ):
        """记录聚类结果详情"""
        self.logger.info("\nClustering Results:")
        
        # 记录每个簇的基本统计信息
        feature_names = [
            'Avg Train Loss',
            'Loss Stability',
            'Loss Trend'
        ]
        
        for cluster_idx in range(2):
            cluster_mask = labels == cluster_idx
            cluster_features = features[cluster_mask]
            cluster_clients = np.array(client_ids)[cluster_mask]
            
            if len(cluster_features) > 0:
                self.logger.info(f"\nCluster {cluster_idx} "
                               f"({'Noisy' if cluster_idx == noisy_cluster else 'Clean'}):")
                self.logger.info(f"Number of clients: {len(cluster_features)}")
                self.logger.info(f"Client IDs: {cluster_clients.tolist()}")
                self.logger.info("\nAverage metrics:")
                
                for i, name in enumerate(feature_names):
                    mean_value = np.mean(cluster_features[:, i])
                    std_value = np.std(cluster_features[:, i])
                    self.logger.info(f"- {name}: {mean_value:.4f} ± {std_value:.4f}")

    def _visualize_clustering(
            self,
            round_num: int,
            normalized_features: np.ndarray,
            labels: np.ndarray,
            client_ids: List[int],
            noisy_cluster: int
    ):
        """
        可视化聚类结果
        """
        try:
            # 设置样式
            plt.style.use('seaborn')
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 1. 训练损失vs验证准确率
            scatter1 = ax1.scatter(
                normalized_features[:, 0],  # 训练损失
                normalized_features[:, 1],  # 验证准确率
                c=labels,
                cmap='coolwarm',
                s=100
            )
            
            # 添加客户端ID标签
            for i, txt in enumerate(client_ids):
                ax1.annotate(f'Client {txt}', 
                           (normalized_features[i, 0], normalized_features[i, 1]))
            
            ax1.set_xlabel('Normalized Training Loss')
            ax1.set_ylabel('Normalized Validation Accuracy')
            ax1.set_title('Client Clustering: Loss vs Accuracy')
            
            # 2. 稳定性vs趋势
            scatter2 = ax2.scatter(
                normalized_features[:, 2],  # 损失稳定性
                normalized_features[:, 4],  # 损失趋势
                c=labels,
                cmap='coolwarm',
                s=100
            )
            
            for i, txt in enumerate(client_ids):
                ax2.annotate(f'Client {txt}', 
                           (normalized_features[i, 2], normalized_features[i, 4]))
            
            ax2.set_xlabel('Normalized Loss Stability')
            ax2.set_ylabel('Normalized Loss Trend')
            ax2.set_title('Client Clustering: Stability vs Trend')
            
            # 添加图例
            legend1 = ax1.legend(*scatter1.legend_elements(),
                               title="Clusters",
                               labels=['Clean', 'Noisy'] if noisy_cluster == 1 else ['Noisy', 'Clean'])
            ax1.add_artist(legend1)
            
            legend2 = ax2.legend(*scatter2.legend_elements(),
                               title="Clusters",
                               labels=['Clean', 'Noisy'] if noisy_cluster == 1 else ['Noisy', 'Clean'])
            ax2.add_artist(legend2)
            
            plt.tight_layout()
            
            # 保存图形
            save_path = os.path.join(self.save_dir, f'clustering_round_{round_num}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved clustering visualization to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing clustering results: {str(e)}")

    def get_detection_stats(self) -> Dict:
        """
        获取检测统计信息，只关注训练损失指标

        Returns:
            Dict: 检测统计信息字典
        """
        try:
            stats = {
                'total_clients': len(self.client_history),
                'noisy_clients': len(self.noisy_clients),
                'noisy_client_ids': list(self.noisy_clients)
            }

            # 计算clean和noisy客户端的平均指标
            if self.client_history:
                recent_metrics = {
                    client_id: metrics[-1]
                    for client_id, metrics in self.client_history.items()
                    if metrics
                }

                clean_metrics = []
                noisy_metrics = []

                for client_id, metrics in recent_metrics.items():
                    metrics_dict = {
                        'train_loss': metrics['train_loss']
                    }

                    if client_id in self.noisy_clients:
                        noisy_metrics.append(metrics_dict)
                    else:
                        clean_metrics.append(metrics_dict)

                if clean_metrics:
                    stats['clean_clients'] = {
                        'avg_train_loss': np.mean([m['train_loss'] for m in clean_metrics]),
                        'std_train_loss': np.std([m['train_loss'] for m in clean_metrics]),
                        'num_clients': len(clean_metrics)
                    }

                if noisy_metrics:
                    stats['noisy_clients'] = {
                        'avg_train_loss': np.mean([m['train_loss'] for m in noisy_metrics]),
                        'std_train_loss': np.std([m['train_loss'] for m in noisy_metrics]),
                        'num_clients': len(noisy_metrics)
                    }

                # 添加聚类性能指标
                if self.clustering_history:
                    latest_clustering = self.clustering_history[-1]
                    stats['clustering'] = {
                        'round': latest_clustering['round'],
                        'num_samples': len(latest_clustering['client_ids']),
                        'cluster_sizes': [
                            np.sum(latest_clustering['labels'] == 0),
                            np.sum(latest_clustering['labels'] == 1)
                        ],
                        'cluster_centers': latest_clustering['cluster_centers']
                    }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting detection stats: {str(e)}")
            # 如果出错，返回包含必要字段的基本统计信息
            return {
                'total_clients': len(self.client_history),
                'noisy_clients': len(self.noisy_clients),
                'clean_clients': {
                    'avg_train_loss': 0.0,
                    'std_train_loss': 0.0,
                    'num_clients': 0
                },
                'noisy_clients': {
                    'avg_train_loss': 0.0,
                    'std_train_loss': 0.0,
                    'num_clients': 0
                }
            }