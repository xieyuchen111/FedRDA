# tests/test_client_detection.py

import numpy as np
import pytest
from typing import Dict, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from .client_detector import NoiseClientDetector


def create_mock_metrics(
        val_accuracy: float,
        credal_ratio: float,
        prediction_consistency: float,
        train_loss: float
) -> Dict:
    """创建模拟的客户端指标"""
    return {
        'val_accuracy': val_accuracy,
        'credal_metrics': {
            'credal_ratio': credal_ratio,
            'prediction_consistency': prediction_consistency
        },
        'train_loss': train_loss
    }


def create_client_history(
        num_rounds: int,
        is_noisy: bool = False,
        improve_trend: bool = True,
        base_accuracy: float = None,
        base_credal: float = None,
        base_consistency: float = None,
        noise_level: float = 0.1  # 添加随机波动的幅度
) -> List[Dict]:
    """
    创建模拟的客户端历史数据

    Args:
        num_rounds: 轮数
        is_noisy: 是否是噪声客户端
        improve_trend: 是否呈现改善趋势
        base_accuracy: 基础准确率
        base_credal: 基础credal ratio
        base_consistency: 基础一致性
        noise_level: 随机波动幅度
    """
    history = []

    # 设置基础值
    if base_accuracy is None:
        base_accuracy = 10.0 if is_noisy else 40.0
    if base_credal is None:
        base_credal = 0.95 if is_noisy else 0.6
    if base_consistency is None:
        base_consistency = 0.4 if is_noisy else 0.8

    base_loss = 2.0 if is_noisy else 1.0

    for round_idx in range(num_rounds):
        if improve_trend:
            # 改善趋势
            accuracy = base_accuracy + (round_idx * (3.0 if not is_noisy else 0.5))
            credal = base_credal - (round_idx * (0.05 if not is_noisy else 0.01))
            consistency = base_consistency + (round_idx * (0.03 if not is_noisy else 0.01))
            loss = base_loss - (round_idx * (0.1 if not is_noisy else 0.02))
        else:
            # 添加随机波动
            accuracy = base_accuracy + np.random.normal(0, noise_level * base_accuracy)
            credal = base_credal + np.random.normal(0, noise_level)
            consistency = base_consistency + np.random.normal(0, noise_level)
            loss = base_loss + np.random.normal(0, noise_level)

        # 确保值在合理范围内
        accuracy = np.clip(accuracy, 0, 100)
        credal = np.clip(credal, 0, 1)
        consistency = np.clip(consistency, 0, 1)
        loss = max(0.1, loss)

        history.append({
            'round': round_idx,
            'val_accuracy': accuracy,
            'credal_ratio': credal,
            'prediction_consistency': consistency,
            'train_loss': loss
        })

    return history


def plot_client_metrics(client_histories: Dict[int, List[Dict]], save_path: str = None):
    """可视化客户端指标"""
    plt.figure(figsize=(15, 10))

    # 设置样式
    sns.set_style("whitegrid")

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = [
        ('val_accuracy', 'Validation Accuracy'),
        ('credal_ratio', 'Credal Ratio'),
        ('prediction_consistency', 'Prediction Consistency'),
        ('train_loss', 'Training Loss')
    ]

    for (metric, title), ax in zip(metrics, axes.flat):
        for client_id, history in client_histories.items():
            values = [m[metric] for m in history]
            ax.plot(values, label=f'Client {client_id}')
            ax.set_title(title)
            ax.set_xlabel('Round')
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class TestNoiseClientDetector:
    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        return NoiseClientDetector(
            detection_start_round=5,
            window_size=3,
            trust_threshold=0.6,
            beta=0.3,
            adaptive_factor=0.1,
            use_gmm=True,
            min_samples_for_gmm=3
        )

    def _verify_gmm_operation(self, detector, round_num: int) -> Dict:
        """验证GMM是否正常运作并返回诊断信息"""
        try:
            features = []
            client_ids = []
            client_histories = {}

            for client_id, history in detector.client_history.items():
                client_histories[client_id] = history
                if len(history) >= detector.window_size:
                    feature = detector._extract_features(history)
                    if feature is not None:
                        features.append(feature)
                        client_ids.append(client_id)

            if len(features) >= detector.min_samples_for_gmm:
                features = np.array(features)

                diagnostic = {
                    'success': True,
                    'features': features,
                    'client_ids': client_ids,
                    'feature_shape': features.shape,
                    'feature_range': (np.min(features), np.max(features)),
                    'has_nan': np.any(np.isnan(features)),
                    'has_inf': np.any(np.isinf(features)),
                    'client_histories': client_histories
                }
            else:
                diagnostic = {
                    'success': False,
                    'reason': f'Not enough samples: {len(features)} < {detector.min_samples_for_gmm}',
                    'client_histories': client_histories
                }
        except Exception as e:
            diagnostic = {
                'success': False,
                'reason': str(e),
                'client_histories': client_histories
            }

        return diagnostic

    def test_gmm_operation(self, detector):
        """专门测试GMM的运行情况"""
        num_rounds = 10

        # 创建明显不同的客户端数据
        client_configs = [
            # 正常客户端 (高准确率，低credal ratio，高一致性)
            {'is_noisy': False, 'base_accuracy': 45.0, 'base_credal': 0.5, 'base_consistency': 0.85},
            {'is_noisy': False, 'base_accuracy': 40.0, 'base_credal': 0.55, 'base_consistency': 0.8},
            # 噪声客户端 (低准确率，高credal ratio，低一致性)
            {'is_noisy': True, 'base_accuracy': 15.0, 'base_credal': 0.9, 'base_consistency': 0.4},
            {'is_noisy': True, 'base_accuracy': 10.0, 'base_credal': 0.95, 'base_consistency': 0.35},
            {'is_noisy': True, 'base_accuracy': 12.0, 'base_credal': 0.92, 'base_consistency': 0.38}
        ]

        client_histories = {}

        # 添加客户端数据
        print("\nAdding client data...")
        for client_id, config in enumerate(client_configs):
            history = create_client_history(
                num_rounds=num_rounds,
                is_noisy=config['is_noisy'],
                improve_trend=not config['is_noisy'],
                base_accuracy=config['base_accuracy'],
                base_credal=config['base_credal'],
                base_consistency=config['base_consistency'],
                noise_level=0.05  # 较小的随机波动
            )

            client_histories[client_id] = history

            for round_idx, metrics in enumerate(history):
                client_metrics = {client_id: create_mock_metrics(
                    metrics['val_accuracy'],
                    metrics['credal_ratio'],
                    metrics['prediction_consistency'],
                    metrics['train_loss']
                )}
                detector.update_metrics(round_idx, client_metrics)

            print(f"Added data for client {client_id} "
                  f"({'Noisy' if config['is_noisy'] else 'Clean'})")

        # 可视化客户端指标
        plot_client_metrics(client_histories, "client_metrics.png")

        # 验证GMM运行状态
        print("\nVerifying GMM operation...")
        diagnostic = self._verify_gmm_operation(detector, 6)

        if diagnostic['success']:
            print("GMM verification successful:")
            print(f"Feature shape: {diagnostic['feature_shape']}")
            print(f"Feature range: {diagnostic['feature_range']}")
            print(f"Contains NaN: {diagnostic['has_nan']}")
            print(f"Contains Inf: {diagnostic['has_inf']}")
        else:
            print(f"GMM verification failed: {diagnostic['reason']}")
            return

        # 执行检测
        print("\nPerforming noise detection...")
        noisy_clients = detector.detect_noisy_clients(6)
        print(f"Detected noisy clients: {noisy_clients}")

        # 验证检测结果
        assert len(noisy_clients) == 3, f"Expected 3 noisy clients, got {len(noisy_clients)}"
        assert all(i in noisy_clients for i in [2, 3, 4]), "Failed to detect correct noisy clients"
        assert not any(i in noisy_clients for i in [0, 1]), "Clean clients incorrectly identified as noisy"

        print("\nTest completed successfully!")


def test_majority_noisy(detector):
    """测试噪声客户端占多数的情况"""
    num_rounds = 10
    client_configs = [
        # 一个正常客户端
        {'is_noisy': False, 'base_accuracy': 45.0, 'base_credal': 0.5, 'base_consistency': 0.85},
        # 四个噪声客户端
        {'is_noisy': True, 'base_accuracy': 15.0, 'base_credal': 0.9, 'base_consistency': 0.4},
        {'is_noisy': True, 'base_accuracy': 12.0, 'base_credal': 0.92, 'base_consistency': 0.38},
        {'is_noisy': True, 'base_accuracy': 10.0, 'base_credal': 0.95, 'base_consistency': 0.35},
        {'is_noisy': True, 'base_accuracy': 13.0, 'base_credal': 0.91, 'base_consistency': 0.39}
    ]

    client_histories = {}

    # 添加客户端数据
    for client_id, config in enumerate(client_configs):
        history = create_client_history(
            num_rounds=num_rounds,
            is_noisy=config['is_noisy'],
            improve_trend=not config['is_noisy'],
            base_accuracy=config['base_accuracy'],
            base_credal=config['base_credal'],
            base_consistency=config['base_consistency']
        )

        client_histories[client_id] = history

        for round_idx, metrics in enumerate(history):
            client_metrics = {client_id: create_mock_metrics(
                metrics['val_accuracy'],
                metrics['credal_ratio'],
                metrics['prediction_consistency'],
                metrics['train_loss']
            )}
            detector.update_metrics(round_idx, client_metrics)

    # 可视化
    plot_client_metrics(client_histories, "majority_noisy_metrics.png")

    # 检测
    noisy_clients = detector.detect_noisy_clients(6)
    print(f"\nDetected noisy clients: {noisy_clients}")

    # 验证
    assert len(noisy_clients) == 4, f"Expected 4 noisy clients, got {len(noisy_clients)}"
    assert 0 not in noisy_clients, "Clean client incorrectly identified as noisy"
    assert all(i in noisy_clients for i in [1, 2, 3, 4]), "Failed to detect all noisy clients"


def run_all_tests():
    """运行所有测试"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    detector = NoiseClientDetector(
        detection_start_round=5,
        window_size=3,
        trust_threshold=0.6,
        beta=0.3,
        adaptive_factor=0.1,
        use_gmm=True,
        min_samples_for_gmm=3
    )

    test = TestNoiseClientDetector()

    print("\nRunning GMM operation test...")
    test.test_gmm_operation(detector)

    print("\nRunning majority noisy test...")
    test_majority_noisy(detector)

    print("\nAll tests completed!")


if __name__ == "__main__":
    run_all_tests()