# utils/transforms.py

import torch
import torchvision.transforms as transforms

class ThreeCropsTransform:
    """
    生成三个视图的转换:
    - 两个强增强视图用于对比学习
    - 一个弱增强视图用于分类
    """
    def __init__(self, strong_transform, weak_transform):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        crops = []
        # 两个强增强视图
        crops.append(self.strong_transform(x))
        crops.append(self.strong_transform(x))
        # 一个弱增强视图
        crops.append(self.weak_transform(x))
        return crops

def get_transform(args):
    """获取数据增强策略"""
    # 强增强 - 用于对比学习分支
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 随机颜色抖动
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),  # 随机灰度
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 弱增强 - 用于分类分支
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform = ThreeCropsTransform(
        strong_transform=strong_transform,
        weak_transform=weak_transform
    )

    return train_transform, test_transform

class GaussianBlur:
    """高斯模糊 - 为指静脉图像特别添加"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_fingervein_transform(args):
    """指静脉数据集的特殊增强策略"""
    # 强增强
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 添加垂直翻转
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),  # 添加高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 弱增强
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 添加垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform = ThreeCropsTransform(
        strong_transform=strong_transform,
        weak_transform=weak_transform
    )

    return train_transform, test_transform