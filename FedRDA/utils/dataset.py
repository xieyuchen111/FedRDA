#utils/dataset.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, target, index, original_target = self.dataset[idx]

        if self.transform:
            # 转换为PIL格式以应用transform
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            # 生成三个视图
            im_aug1 = self.transform.strong_transform(img)
            im_aug2 = self.transform.strong_transform(img)
            im_weak = self.transform.weak_transform(img)
            images = (im_aug1, im_aug2, im_weak)
        else:
            images = (img, img, img)

        return (images, target, index, original_target)

    def __len__(self):
        return len(self.dataset)

class FingerVeinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        指静脉数据集
        Args:
            root_dir: 数据集根目录
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有类别
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有图像路径和标签，每个类限制6个样本
        self.image_paths = []
        self.targets = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            # 获取并排序该类别下的所有图像文件
            img_files = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])

            # 只取前6个样本
            img_files = img_files[:6]
            cls_idx = self.class_to_idx[cls_name]

            # 添加图像路径和标签
            for img_name in img_files:
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.targets.append(cls_idx)

        self.targets = np.array(self.targets)
        self.num_classes = len(self.classes)
        self.num_samples = len(self.targets)

        # 验证每个类别是否都有6个样本
        class_counts = {}
        for target in self.targets:
            class_counts[target] = class_counts.get(target, 0) + 1

        for cls_idx, count in class_counts.items():
            assert count == 6, f"Class {cls_idx} has {count} samples, expected 6"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, idx, label  # 返回 image, target, idx, original_target


# class FingerVeinDataset(Dataset):
#     def __init__(self, root_dir, noise_ratio=0.3, train=True, transform=None, random_state=42):
#         self.root_dir = root_dir
#         self.train = train
#         self.transform = transform
#         self.noise_ratio = noise_ratio
#         self.random_state = random_state

#         # 获取所有类别
#         self.classes = sorted(os.listdir(root_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

#         # 收集所有图像路径和标签
#         self.image_paths = []
#         self.targets = []
#         self.class_samples = {i: 0 for i in range(len(self.classes))}

#         for cls_name in self.classes:
#             cls_dir = os.path.join(root_dir, cls_name)
#             if not os.path.isdir(cls_dir):
#                 continue
#             cls_idx = self.class_to_idx[cls_name]
#             for img_name in os.listdir(cls_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                     self.image_paths.append(os.path.join(cls_dir, img_name))
#                     self.targets.append(cls_idx)
#                     self.class_samples[cls_idx] += 1

#         self.targets = np.array(self.targets)
#         self.num_classes = len(self.classes)
#         self.num_samples = len(self.targets)

#         # 添加噪声
#         if self.train:
#             np.random.seed(self.random_state)
#             self.original_targets = self.targets.copy()
#             self._add_noise()
#         else:
#             self.noisy_indices = []
#             self.original_targets = self.targets.copy()

#     def _add_noise(self):
#         """添加噪声并确保每个类别至少保留一个干净样本"""
#         num_noisy = int(self.noise_ratio * self.num_samples)
#         all_indices = np.arange(self.num_samples)
        
#         # 确保每个类别至少保留一个干净样本
#         reserved_indices = []
#         for class_idx in range(self.num_classes):
#             class_indices = np.where(self.targets == class_idx)[0]
#             if len(class_indices) > 0:
#                 reserved_indices.append(np.random.choice(class_indices))
        
#         # 从剩余样本中选择噪声样本
#         available_indices = list(set(all_indices) - set(reserved_indices))
#         num_noisy = min(num_noisy, len(available_indices))
#         self.noisy_indices = np.random.choice(available_indices, num_noisy, replace=False)

#         for idx in self.noisy_indices:
#             current_label = self.targets[idx]
#             possible_labels = list(range(self.num_classes))
#             possible_labels.remove(current_label)
#             new_label = np.random.choice(possible_labels)
#             self.targets[idx] = new_label

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             image = Image.new('RGB', (224, 224))

#         label = self.targets[idx]
#         original_label = self.original_targets[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label, idx, original_label

# def get_data_loaders(args):
#     """创建数据加载器"""
#     # 数据增强
#     transform_train = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     transform_test = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 创建数据集
#     train_dataset = FingerVeinDataset(
#         root_dir=args.data_root,
#         noise_ratio=args.noise_ratio,
#         train=True,
#         transform=transform_train
#     )

#     val_dataset = FingerVeinDataset(
#         root_dir=args.data_root,
#         train=False,
#         transform=transform_test
#     )

#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )

#     return train_loader, val_loader