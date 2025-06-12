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
        self.confidence_threshold = confidence_threshold
        
        # 添加缓存和统计
        self.cached_soft_labels = None
        self.cached_noisy_indices = None
        self.valid_soft_label_count = 0
        self.total_noisy_samples = 0
        self.round_statistics = {}
        
        # 聚类参数
        self.n_clusters = 2
        self.random_state = 42
        # 添加新的属性
        self.soft_label_momentum = 0.8  # 软标签动量系数
        self.update_frequency = 3  # 每隔几个epoch更新一次
        self.entropy_threshold = 0.5  # 熵阈值
        self.last_update_epoch = 0  # 上次更新的epoch
        self.label_history = {}  # 存储标签历史

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

    def update_centers_and_statistics(self):
        """更新类别中心和计算统计信息"""
        logging.info("\nUpdating class centers and computing statistics...")
        clean_samples = set(self.feature_bank.keys()) - self.noisy_samples
        
        # 重置类别中心和方差
        self.class_centers.clear()
        self.class_vars.clear()
        
        class_stats = defaultdict(dict)
        
        for class_id in range(self.num_classes):
            class_features = []
            for idx in self.class_samples[class_id]:
                if idx in clean_samples and idx in self.feature_bank:
                    class_features.append(self.feature_bank[idx])
            
            if class_features:
                class_features = np.stack(class_features)
                self.class_centers[class_id] = np.mean(class_features, axis=0)
                self.class_vars[class_id] = np.var(class_features, axis=0) + 1e-6
                
                # 记录统计信息
                class_stats[class_id] = {
                    'clean_samples': len(class_features),
                    'feature_mean': np.mean(class_features),
                    'feature_std': np.std(class_features)
                }
        
        # logging.info(f"Updated centers for {len(self.class_centers)} classes")
        # for class_id, stats in class_stats.items():
        #     logging.info(f"Class {class_id}: {stats['clean_samples']} clean samples, "
        #               f"mean={stats['feature_mean']:.3f}, std={stats['feature_std']:.3f}")

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

    def _compute_soft_labels_with_model_eval(self, model=None):
        """使用模型评估模式计算软标签（批处理版本）"""
        logging.info("\nComputing soft labels with model evaluation...")
        if model is not None:
            model.eval()
    
        soft_labels = {}
        valid_count = 0
        skipped_count = 0
        confidence_stats = []
        entropy_stats = []
    
        # 处理干净样本
        clean_indices = set(self.loss_history.keys()) - self.noisy_samples
        for idx in clean_indices:
            soft_labels[idx] = {'soft_label': None, 'confidence': 1.0}
    
        # 批处理噪声样本
        batch_size = 64
        noisy_samples = list(self.noisy_samples)
        
        for i in range(0, len(noisy_samples), batch_size):
            batch_indices = noisy_samples[i:i + batch_size]
            batch_features = []
            valid_idxs = []
    
            # 收集有效样本
            for idx in batch_indices:
                if idx in self.feature_bank:
                    batch_features.append(self.feature_bank[idx])
                    valid_idxs.append(idx)
    
            if not batch_features:
                continue
    
            # 批量处理模型预测
            with torch.no_grad():
                batch_history = [self.loss_history[idx][-1] for idx in valid_idxs]
                batch_logits = torch.stack([
                    torch.from_numpy(h['logits']).to(self.device) 
                    for h in batch_history
                ])
                batch_probs = F.softmax(batch_logits / self.temperature, dim=1)
                
                # 计算预测熵
                entropy = -torch.sum(batch_probs * torch.log(batch_probs + 1e-6), dim=1)
                normalized_entropy = entropy / np.log(self.num_classes)
                
                # 获取最高预测概率
                max_probs, _ = torch.max(batch_probs, dim=1)
    
                # 基于熵的确定性判断
                for idx, probs, ent, max_prob in zip(
                    valid_idxs, batch_probs, normalized_entropy, max_probs
                ):
                    # 根据当前轮次动态调整熵阈值
                    current_entropy_threshold = self.entropy_threshold
                    if self.current_round < 15:
                        # 前期使用更宽松的熵阈值
                        current_entropy_threshold = 0.7
                    
                    if ent < current_entropy_threshold:
                        if idx in self.label_history:
                            # 使用动量更新
                            old_label = self.label_history[idx]['soft_label']
                            new_label = probs.cpu().numpy()
                            soft_label = (self.soft_label_momentum * old_label + 
                                        (1 - self.soft_label_momentum) * new_label)
                            
                            # 归一化确保概率和为1
                            soft_label = soft_label / np.sum(soft_label)
                        else:
                            soft_label = probs.cpu().numpy()
                        
                        soft_labels[idx] = {
                            'soft_label': soft_label,
                            'confidence': max_prob.item(),
                            'entropy': ent.item()
                        }
                        valid_count += 1
                        confidence_stats.append(max_prob.item())
                        entropy_stats.append(ent.item())
                    else:
                        soft_labels[idx] = {
                            'soft_label': None, 
                            'confidence': 0.0,
                            'entropy': ent.item()
                        }
                        skipped_count += 1
    
        # 更新标签历史
        self.label_history.update({
            k: v for k, v in soft_labels.items() 
            if v['soft_label'] is not None
        })
    
        # 记录统计信息
        self.valid_soft_label_count = valid_count
        if confidence_stats:
            mean_confidence = np.mean(confidence_stats)
            std_confidence = np.std(confidence_stats)
            mean_entropy = np.mean(entropy_stats)
            std_entropy = np.std(entropy_stats)
        else:
            mean_confidence = mean_entropy = 0
            std_confidence = std_entropy = 0
    
        logging.info(f"Soft label computation complete:")
        logging.info(f"Valid soft labels: {valid_count}")
        logging.info(f"Skipped samples: {skipped_count}")
        logging.info(f"Mean confidence: {mean_confidence:.3f} ± {std_confidence:.3f}")
        logging.info(f"Mean entropy: {mean_entropy:.3f} ± {std_entropy:.3f}")
    
        return soft_labels

    def detect_noisy_samples(self, is_noisy_client=False, model=None):
        """检测噪声样本并生成软标签"""
        logging.info(f"\nStarting noise detection for round {self.current_round}")
        current_local_epoch = getattr(model, 'current_local_epoch', 1)
    
        # 如果还未到检测轮次，返回默认值
        if self.current_round < self.detection_round:
            logging.info("Not yet reached detection round, returning default values")
            return {idx: {'soft_label': None, 'confidence': 1.0}
                    for idx in self.loss_history.keys()}, []
    
        # 第一次达到检测轮次时，执行初始检测
        if self.current_round == self.detection_round and not self.detection_completed:
            logging.info("Initial noise detection round")
            try:
                noisy_samples = []
                class_noise_counts = defaultdict(int)
    
                # 按类别处理样本
                for class_id, class_indices in self.class_samples.items():
                    if len(class_indices) < 3:  
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
                self.total_noisy_samples = len(self.noisy_samples)
                
                # 更新类别中心
                self.update_centers_and_statistics()
    
                # 检查是否需要更新软标签
                should_update = (
                    current_local_epoch == 1 or  
                    (current_local_epoch - self.last_update_epoch) >= self.update_frequency
                )
    
                if should_update:
                    logging.info(f"Updating soft labels at epoch {current_local_epoch}")
                    self.last_update_epoch = current_local_epoch
                    soft_labels = self._compute_soft_labels_with_model_eval(model)
                    self.cached_soft_labels = soft_labels
                    
                    # 记录本轮统计信息
                    self.round_statistics[self.current_round] = {
                        'total_noisy': self.total_noisy_samples,
                        'valid_soft_labels': self.valid_soft_label_count,
                        'class_noise_counts': dict(class_noise_counts)
                    }
                    
                    # 记录初始检测结果
                    self._log_detection_results(class_noise_counts, class_features)
                else:
                    logging.info(f"Keeping existing soft labels at epoch {current_local_epoch}")
                    soft_labels = self.cached_soft_labels
    
                return soft_labels, list(self.noisy_samples)
                    
            except Exception as e:
                logging.error(f"Error in initial noise detection: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                return {idx: {'soft_label': None, 'confidence': 1.0}
                        for idx in self.loss_history.keys()}, []
        
        # 后续轮次的处理
        if self.detection_completed:
            logging.info(f"Processing detection for round {self.current_round}")
            
            # 检查是否需要更新软标签
            should_update = (
                current_local_epoch == 1 or  
                (current_local_epoch - self.last_update_epoch) >= self.update_frequency
            )
            
            if should_update:
                logging.info(f"Updating soft labels at epoch {current_local_epoch}")
                self.last_update_epoch = current_local_epoch
                
                # 更新类别中心
                self.update_centers_and_statistics()
                
                # 重新计算软标签
                self.cached_soft_labels = self._compute_soft_labels_with_model_eval(model)
                
                # 更新统计信息
                self.round_statistics[self.current_round] = {
                    'total_noisy': self.total_noisy_samples,
                    'valid_soft_labels': self.valid_soft_label_count
                }
                
                # 记录对比信息
                if self.current_round > self.detection_round:
                    prev_valid = self.round_statistics[self.current_round - 1]['valid_soft_labels']
                    curr_valid = self.valid_soft_label_count
                    change = curr_valid - prev_valid
                    logging.info(f"Soft label count change: {change:+d} "
                               f"({prev_valid} -> {curr_valid})")
            else:
                logging.info(f"Keeping existing soft labels at epoch {current_local_epoch}")
            # 确保返回的 noisy_indices 永远是一个列表
            if self.cached_noisy_indices is None:
                self.cached_noisy_indices = list(self.noisy_samples) if self.noisy_samples else []
                
            return self.cached_soft_labels, self.cached_noisy_indices
                
        return self.cached_soft_labels, self.cached_noisy_indices
    def _log_detection_results(self, class_noise_counts, class_features):
        """增强的检测结果日志"""
        logging.info(f"\nNoise Detection Results (Round {self.current_round}):")
        logging.info(f"Total samples processed: {len(self.loss_history)}")
        logging.info(f"Total detected noisy samples: {len(self.noisy_samples)}")

        # for class_id, noise_count in class_noise_counts.items():
        #     total_samples = len(self.class_samples[class_id])
        #     noise_ratio = noise_count / total_samples * 100
        #     logging.info(f"\nClass {class_id}:")
        #     logging.info(f"- Total samples: {total_samples}")
        #     logging.info(f"- Noisy samples: {noise_count}")
        #     logging.info(f"- Noise ratio: {noise_ratio:.1f}%")

        # 整体统计
        total_samples = sum(len(samples) for samples in self.class_samples.values())
        total_noise_ratio = len(self.noisy_samples) / total_samples * 100
        logging.info(f"\nOverall statistics:")
        logging.info(f"- Total samples: {total_samples}")
        logging.info(f"- Overall noise ratio: {total_noise_ratio:.1f}%")

    def get_detection_stats(self):
        """获取检测统计信息"""
        stats = {
            'current_round': self.current_round,
            'total_noisy_samples': self.total_noisy_samples,
            'valid_soft_labels': self.valid_soft_label_count,
            'round_statistics': self.round_statistics
        }
        return stats