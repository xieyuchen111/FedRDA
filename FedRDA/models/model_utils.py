import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """
    保存模型检查点
    Args:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_filepath)
        logging.info(f"Saved best model to {best_filepath}")

def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    加载模型检查点
    Args:
        model: 模型实例
        checkpoint_path: 检查点路径
        optimizer: 优化器实例（可选）
    Returns:
        epoch: 训练轮次
        best_acc: 最佳准确率
    """
    if not os.path.exists(checkpoint_path):
        return 0, 0.0
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    return epoch, best_acc

def accuracy(output, target, topk=(1,)):
    """
    计算前k个预测的准确率
    Args:
        output: 模型输出
        target: 目标标签
        topk: 要计算的top-k准确率
    Returns:
        list: top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_model(model, data_loader, criterion, device, detailed_metrics=False):
    """
    评估模型性能
    Args:
        model: 模型实例
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        detailed_metrics: 是否计算详细指标
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, labels, _, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels).mean()
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # 计算指标
    predictions = np.array(predictions)
    targets = np.array(targets)
    accuracy = np.mean(predictions == targets) * 100
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy
    }
    
    if detailed_metrics:
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(targets, predictions)
        # 计算每个类别的指标
        report = classification_report(targets, predictions, output_dict=True)
        
        metrics.update({
            'confusion_matrix': conf_matrix,
            'classification_report': report
        })
    
    return metrics

def get_model_predictions(model, data_loader, device):
    """
    获取模型预测结果
    Args:
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
    Returns:
        predictions: 预测结果
        probabilities: 预测概率
    """
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, _, _, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def calculate_metrics_per_class(predictions, targets, num_classes):
    """
    计算每个类别的指标
    Args:
        predictions: 预测结果
        targets: 目标标签
        num_classes: 类别数量
    Returns:
        dict: 每个类别的指标
    """
    metrics_per_class = {}
    
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predictions[class_mask] == targets[class_mask])
            class_count = np.sum(class_mask)
            metrics_per_class[class_idx] = {
                'accuracy': class_acc * 100,
                'sample_count': class_count
            }
    
    return metrics_per_class