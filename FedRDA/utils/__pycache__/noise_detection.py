# utils/noise_detection.py

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import logging


class DynamicNoiseDetector:
    def __init__(self, num_classes, detection_round=3, noise_ratio=0.5,
                 feature_dim=512, temperature=0.5, confidence_threshold=0.9, device='cuda'):
        """初始化
        Args:
            num_classes: 类别数量
            detection_round: 开始检测的轮次
            noise_ratio: 噪声比例
            feature_dim: 特征维度
            temperature: 软标签温度参数
            confidence_threshold: 预测置信度阈值
        """
        # 基本参数
        self.num_classes = num_classes
        self.detection_round = detection_round + 1
        self.noise_ratio = noise_ratio
        self.loss_history = defaultdict(list)
        self.class_samples = defaultdict(list)
        self.feature_bank = {}
        self.detection_completed = False
        self.noisy_samples = set()
        self.current_round = 0
        self.device = device
        # 特征相关参数
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.class_centers = {}  # 存储类别特征中心
        self.class_vars = {}  # 存储类内特征方差

        # 置信度参数
        self.confidence_threshold = confidence_threshold  # 新增

        # 聚类参数
        self.n_clusters = 2
        self.random_state = 42

    def update_history(self, indices, losses, targets, epoch, features=None, logits=None):
        """更新历史记录
        Args:
            indices: 样本索引
            losses: 损失值
            targets: 标签
            epoch: 当前训练轮次
            features: 样本特征 [B, feature_dim]
            logits: 模型预测logits [B, num_classes]
        """
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else indices
        losses = losses.detach().cpu().numpy() if torch.is_tensor(losses) else losses
        targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets

        if features is not None:
            features = features.detach().cpu().numpy() if torch.is_tensor(features) else features
        if logits is not None:
            logits = logits.detach().cpu().numpy() if torch.is_tensor(logits) else logits

        for i, (idx, loss, target) in enumerate(zip(indices, losses, targets)):
            idx_item = int(idx)
            loss_value = float(loss)
            target_value = int(target)

            # 更新历史记录
            history_item = {
                'epoch': epoch,
                'loss': loss_value,
                'target': target_value
            }
            if logits is not None:
                history_item['logits'] = logits[i]

            self.loss_history[idx_item].append(history_item)

            if features is not None:
                self.feature_bank[idx_item] = features[i]

            if epoch == 1:
                self.class_samples[target_value].append(idx_item)

    def _compute_sample_features(self, idx):
        """简化后的样本特征计算"""
        if len(self.loss_history[idx]) < 2:
            return None

        losses = [h['loss'] for h in self.loss_history[idx]]

        features = {
            'avg_loss': np.mean(losses),
            'loss_std': np.std(losses),
            'loss_trend': losses[-1] - losses[0]
        }

        return features

    def _perform_clustering(self, feature_matrix):
        """执行聚类"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(normalized_features)

        cluster_losses = []
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                avg_loss = np.mean(feature_matrix[cluster_mask, 0])
                cluster_losses.append((i, avg_loss))

        noisy_cluster = max(cluster_losses, key=lambda x: x[1])[0]
        return labels, noisy_cluster

    def _compute_class_centers(self):
        """计算每个类别的特征中心和方差"""
        clean_samples = set(self.feature_bank.keys()) - self.noisy_samples

        for class_id in range(self.num_classes):
            class_features = []
            for idx in self.class_samples[class_id]:
                if idx in clean_samples and idx in self.feature_bank:
                    class_features.append(self.feature_bank[idx])

            if class_features:
                class_features = np.stack(class_features)
                self.class_centers[class_id] = np.mean(class_features, axis=0)
                self.class_vars[class_id] = np.var(class_features, axis=0) + 1e-6

    def _compute_similarity_score(self, feature, class_id):
        """计算特征与类别中心的相似度(GPU版本)"""
        if class_id not in self.class_centers:
            return 0.0

        # 转换为torch张量并移至GPU
        feature = torch.from_numpy(feature).to(self.device)
        center = torch.from_numpy(self.class_centers[class_id]).to(self.device)
        var = torch.from_numpy(self.class_vars[class_id]).to(self.device)

        # 计算余弦相似度
        norm_feature = feature / (torch.norm(feature) + 1e-6)
        norm_center = center / (torch.norm(center) + 1e-6)
        cos_sim = torch.dot(norm_feature, norm_center)

        # 计算标准化的欧氏距离
        normalized_dist = torch.sum(((feature - center) ** 2) / (var + 1e-6)) / self.feature_dim

        # 结合两种相似度
        similarity = (1 + cos_sim) / 2
        similarity *= torch.exp(-normalized_dist * self.temperature)

        return similarity.cpu().item()

    def _compute_soft_labels(self):
        """计算软标签(GPU版本)"""
        soft_labels = {}

        for idx in self.loss_history.keys():
            if idx not in self.noisy_samples:
                soft_labels[idx] = {
                    'soft_label': None,
                    'confidence': 1.0
                }
                continue

            if idx in self.feature_bank:
                feature = self.feature_bank[idx]
                similarities = torch.zeros(self.num_classes, device=self.device)

                # 计算特征相似度
                for class_id in range(self.num_classes):
                    sim = self._compute_similarity_score(feature, class_id)
                    similarities[class_id] = sim

                if torch.sum(similarities) > 0:
                    # 归一化相似度
                    similarities = similarities / torch.sum(similarities)

                    if self.current_round == self.detection_round:
                        soft_labels[idx] = {
                            'soft_label': similarities.cpu().numpy(),
                            'confidence': torch.max(similarities).cpu().item()
                        }
                    else:
                        # 后续轮次结合预测置信度
                        latest_history = self.loss_history[idx][-1]
                        logits = torch.from_numpy(latest_history['logits']).to(self.device)
                        pred_probs = F.softmax(logits / self.temperature, dim=0)
                        pred_conf = torch.max(pred_probs).cpu().item()

                        if pred_conf >= self.confidence_threshold:
                            soft_labels[idx] = {
                                'soft_label': similarities.cpu().numpy(),
                                'confidence': pred_conf
                            }
                        else:
                            soft_labels[idx] = {
                                'soft_label': None,
                                'confidence': 0.0
                            }

        return soft_labels

    def detect_noisy_samples(self, is_noisy_client=False):
        """检测噪声样本并生成软标签"""
        # 如果还未到检测轮次，返回默认值
        if self.current_round != self.detection_round:
            return {idx: {'soft_label': None, 'confidence': 1.0}
                    for idx in self.loss_history.keys()}, []

        # 如果检测已完成，只更新软标签
        if self.detection_completed:
            # 更新类别中心和方差(使用最新特征)
            self._compute_class_centers()
            # 计算并返回最新的软标签
            return self._compute_soft_labels(), list(self.noisy_samples)

        try:
            noisy_samples = []
            class_noise_counts = defaultdict(int)

            # 按类别处理样本
            for class_id, class_indices in self.class_samples.items():
                if len(class_indices) < 3:  # 跳过样本太少的类别
                    continue

                # 1. 收集该类别所有样本的特征
                class_features = {}
                feature_list = []
                valid_indices = []

                for idx in class_indices:
                    features = self._compute_sample_features(idx)
                    if features is not None:
                        class_features[idx] = features
                        feature_vector = [
                            features['avg_loss'],
                            features['loss_std'],
                            features['loss_trend']
                        ]
                        feature_list.append(feature_vector)
                        valid_indices.append(idx)

                if len(feature_list) < 3:
                    continue

                # 2. 执行聚类
                feature_matrix = np.array(feature_list)
                labels, noisy_cluster = self._perform_clustering(feature_matrix)

                # 3. 获取噪声样本
                for i, idx in enumerate(valid_indices):
                    if labels[i] == noisy_cluster:
                        noisy_samples.append(idx)
                        class_noise_counts[class_id] += 1

            # 更新检测状态
            self.noisy_samples = set(noisy_samples)
            self.detection_completed = True

            # 计算类别中心和初始软标签
            self._compute_class_centers()
            soft_labels = self._compute_soft_labels()

            # 记录检测结果
            self._log_detection_results(class_noise_counts, class_features)

            return soft_labels, list(self.noisy_samples)

        except Exception as e:
            logging.error(f"Error in noise detection: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {idx: {'soft_label': None, 'confidence': 1.0}
                    for idx in self.loss_history.keys()}, []

    def _log_detection_results(self, class_noise_counts, class_features):
        """增强的检测结果日志"""
        logging.info(f"\nNoise Detection Results (Round {self.current_round}):")
        logging.info(f"Total samples processed: {len(self.loss_history)}")
        logging.info(f"Total detected noisy samples: {len(self.noisy_samples)}")

        for class_id, noise_count in class_noise_counts.items():
            total_samples = len(self.class_samples[class_id])
            noise_ratio = noise_count / total_samples * 100

            class_losses = [f['avg_loss'] for f in class_features.values()]
            mean_loss = np.mean(class_losses)
            std_loss = np.std(class_losses)

        # 整体统计
        total_samples = sum(len(samples) for samples in self.class_samples.values())
        total_noise_ratio = len(self.noisy_samples) / total_samples * 100
        logging.info(f"\nOverall noise ratio: {total_noise_ratio:.1f}%")
