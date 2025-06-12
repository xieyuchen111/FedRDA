# semantic_clustering.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class SemanticClustering:
    def __init__(
            self,
            args,
            num_classes: int,
            device: str,
            feature_dim: int,
            pretrained_features: torch.Tensor,  # 预训练特征
            pretrained_knn: torch.Tensor,  # 预训练KNN索引
            k_neighbors: int = 3,
            lambda_e: float = 2.0,

    ):
        """语义聚类模块

        Args:
            num_classes: 类别数
            device: 设备
            feature_dim: 特征维度
            pretrained_features: 预训练特征 [N, D]
            pretrained_knn: 预训练的KNN索引 [N, K]
            k_neighbors: KNN中的K值
            lambda_e: 公式(9)中的熵正则化系数
        """
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = args.feature_dim
        self.k_neighbors = args.k_neighbors
        self.lambda_e = args.lambda_e
        # 选择学习率
        if args.noise_ratio > args.scan_noise_thresh:
            self.lr = args.scan_lr_high
        else:
            self.lr = args.scan_lr_low

        # 优化器参数
        self.momentum = args.scan_momentum
        self.weight_decay = args.scan_weight_decay
        self.pretrained_features = pretrained_features.to(device)
        self.neighbor_indices = pretrained_knn.to(device)
        # 数据加载参数
        self.batch_size = args.scan_batch_size
        # 初始化语义关系矩阵
        self.semantic_relations = None

    def E_step(
            self,
            features: torch.Tensor,
            predictions: torch.Tensor,
            old_relations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """E步：更新语义关系矩阵"""
        try:
            N = features.size(0)
            relations = torch.zeros(N, N).to(self.device)

            # 获取预测标签
            try:
                pred_labels = torch.argmax(predictions, dim=1)
            except Exception as pred_error:
                logging.error(f"获取预测标签失败: {str(pred_error)}")
                raise pred_error

            # 计算语义关系
            for i in range(N):
                try:
                    # 使用预训练的KNN索引
                    neighbors = self.neighbor_indices[i]

                    for j in neighbors:
                        try:
                            # 如果预测标签相同
                            if pred_labels[i] == pred_labels[j]:
                                # 计算特征空间的相似度
                                sim_score = F.cosine_similarity(
                                    features[i].unsqueeze(0),
                                    features[j].unsqueeze(0)
                                )

                                # 计算预测分布的一致性
                                pred_sim = F.cosine_similarity(
                                    predictions[i].unsqueeze(0),
                                    predictions[j].unsqueeze(0)
                                )

                                # 综合得分
                                score = (sim_score * pred_sim).item()
                                relations[i, j] = score
                                relations[j, i] = score
                        except Exception as neighbor_error:
                            logging.error(f"处理样本{i}的邻居{j}时出错: {str(neighbor_error)}")
                            continue
                except Exception as sample_error:
                    logging.error(f"处理样本{i}时出错: {str(sample_error)}")
                    continue

            # 归一化关系矩阵
            try:
                relations = F.normalize(relations, p=1, dim=1)
            except Exception as norm_error:
                logging.error(f"关系矩阵归一化失败: {str(norm_error)}")
                raise norm_error

            # 如果有历史关系，做平滑处理
            if old_relations is not None:
                try:
                    alpha = 0.9  # 平滑系数
                    relations = alpha * relations + (1 - alpha) * old_relations
                except Exception as smooth_error:
                    logging.error(f"历史关系平滑处理失败: {str(smooth_error)}")
                    raise smooth_error

            self.semantic_relations = relations
            return relations

        except Exception as e:
            logging.error(f"E步计算失败: {str(e)}")
            raise e

    def M_step(
            self,
            features: torch.Tensor,
            predictions: torch.Tensor,
            relations: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """M步：优化特征表示和分类"""
        try:
            # 计算聚类损失
            try:
                cluster_loss = self._compute_cluster_loss(features, relations)
            except Exception as cluster_error:
                logging.error(f"计算聚类损失失败: {str(cluster_error)}")
                raise cluster_error

            # 计算熵正则化损失
            try:
                entropy_loss = self._compute_entropy_loss(predictions)
            except Exception as entropy_error:
                logging.error(f"计算熵损失失败: {str(entropy_error)}")
                raise entropy_error

            # 总损失
            total_loss = cluster_loss + self.lambda_e * entropy_loss

            loss_dict = {
                'cluster_loss': cluster_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'total_loss': total_loss.item()
            }

            return total_loss, loss_dict

        except Exception as e:
            logging.error(f"M步计算失败: {str(e)}")
            raise e

    def _compute_cluster_loss(
            self,
            features: torch.Tensor,
            relations: torch.Tensor
    ) -> torch.Tensor:
        """计算聚类损失"""
        try:
            # 计算特征相似度矩阵
            try:
                features_norm = F.normalize(features, dim=1)
                sim_matrix = torch.mm(features_norm, features_norm.t())
            except Exception as sim_error:
                logging.error(f"计算特征相似度矩阵失败: {str(sim_error)}")
                raise sim_error

            # 基于语义关系的加权损失
            try:
                loss = -torch.sum(relations * torch.log(torch.clamp(sim_matrix, min=1e-7)))
                loss = loss / (relations.sum() + 1e-7)
            except Exception as loss_error:
                logging.error(f"计算加权损失失败: {str(loss_error)}")
                raise loss_error

            return loss

        except Exception as e:
            logging.error(f"聚类损失计算失败: {str(e)}")
            raise e

    def _compute_entropy_loss(
            self,
            predictions: torch.Tensor
    ) -> torch.Tensor:
        """计算熵正则化损失"""
        try:
            # 计算每个类别的平均预测概率
            try:
                avg_predictions = torch.mean(predictions, dim=0)
            except Exception as avg_error:
                logging.error(f"计算平均预测概率失败: {str(avg_error)}")
                raise avg_error

            # 计算熵损失
            try:
                entropy = torch.sum(
                    avg_predictions * torch.log(avg_predictions * self.num_classes + 1e-7)
                )
            except Exception as entropy_error:
                logging.error(f"计算熵失败: {str(entropy_error)}")
                raise entropy_error

            return entropy

        except Exception as e:
            logging.error(f"熵损失计算失败: {str(e)}")
            raise e

    def update(
            self,
            features: torch.Tensor,
            predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """执行一次EM更新"""
        try:
            # E步：更新语义关系
            relations = self.E_step(
                features,
                predictions,
                self.semantic_relations
            )

            # M步：计算损失
            total_loss, loss_dict = self.M_step(
                features,
                predictions,
                relations
            )

            return total_loss, loss_dict

        except Exception as e:
            logging.error(f"EM更新失败: {str(e)}")
            raise e

    def get_semantic_relations(self) -> torch.Tensor:
        """获取当前的语义关系矩阵

        Returns:
            torch.Tensor: 语义关系矩阵
        """
        if self.semantic_relations is None:
            raise ValueError("Semantic relations not initialized! Run E-step first.")
        return self.semantic_relations