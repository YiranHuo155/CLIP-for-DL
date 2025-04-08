import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
from config import DATA_PATH, MODEL_CONFIG, AUGMENTATION_CONFIG, DEVICE
import torch.nn.functional as F

def process_multiple_labels(problems_column):
    """处理多个诊断标签
    
    Args:
        problems_column: 包含诊断信息的列
        
    Returns:
        list: 所有唯一的诊断标签列表
    """
    all_labels = []
    for problems in problems_column:
        if isinstance(problems, str):
            # 使用分号分割多个诊断
            labels = [p.strip() for p in problems.split(';')]
            all_labels.extend(labels)
    return list(set(all_labels))

def get_data_transforms():
    """获取数据转换
    
    Returns:
        train_transform: 训练数据转换
        val_transform: 验证数据转换
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_degrees']),
        transforms.RandomAffine(
            degrees=0,
            translate=AUGMENTATION_CONFIG['translate']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION_CONFIG['normalize_mean'],
            std=AUGMENTATION_CONFIG['normalize_std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION_CONFIG['normalize_mean'],
            std=AUGMENTATION_CONFIG['normalize_std']
        )
    ])
    
    return train_transform, val_transform

def preprocess_image(image_path, image_size):
    """预处理图像
    
    Args:
        image_path: 图像路径
        image_size: 目标图像大小
        
    Returns:
        处理后的图像
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 调整大小
        img = cv2.resize(img, (image_size, image_size))
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

class ChestXrayDataset(Dataset):
    """胸部X光数据集类"""
    
    def __init__(self, image_paths, labels, image_size=224, transform=None):
        """
        Args:
            image_paths (list): 图像路径列表
            labels (list): 标签列表
            image_size (int): 图像大小
            transform: 数据转换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(DATA_PATH['image_dir'], self.image_paths[idx])
        try:
            image = preprocess_image(image_path, self.image_size)
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # 返回一个空图像和标签
            return torch.zeros((3, self.image_size, self.image_size)), self.labels[idx]

def prepare_data(test_size=0.2, random_state=42):
    """准备数据集
    
    Args:
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (训练数据, 验证数据)
    """
    # 读取数据
    reports_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['reports_csv'])
    projections_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['projections_csv'])
    
    print(f"读取报告数据：{reports_path}")
    print(f"读取投影数据：{projections_path}")
    
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)
    
    # 合并数据集，使用uid关联
    df = pd.merge(reports_df, projections_df, on='uid')
    
    # 处理多标签
    all_labels = process_multiple_labels(df['Problems'])
    print(f"找到 {len(all_labels)} 个唯一诊断标签")
    
    # 创建标签到索引的映射
    label2idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # 将多标签转换为二进制向量
    label_vectors = []
    for problems in df['Problems']:
        vector = np.zeros(len(all_labels))
        if isinstance(problems, str):
            labels = [l.strip() for l in problems.split(';')]
            for label in labels:
                if label in label2idx:
                    vector[label2idx[label]] = 1
        label_vectors.append(vector)
    
    # 分割数据集
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=test_size,
        random_state=random_state
    )
    
    # 保存训练集和验证集
    train_data = {
        'image_path': df.iloc[train_idx]['filename'].values,
        'labels': [label_vectors[i] for i in train_idx]
    }
    val_data = {
        'image_path': df.iloc[val_idx]['filename'].values,
        'labels': [label_vectors[i] for i in val_idx]
    }
    
    # 保存为CSV
    train_csv_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['train_data'])
    val_csv_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['val_data'])
    
    pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
    pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
    
    print(f"保存训练集到 {train_csv_path}")
    print(f"保存验证集到 {val_csv_path}")
    
    return train_data, val_data, all_labels

def load_data():
    """加载已处理的数据集
    
    Returns:
        tuple: (训练数据, 验证数据, 标签列表)
    """
    train_csv_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['train_data'])
    val_csv_path = os.path.join(DATA_PATH['base_dir'], DATA_PATH['val_data'])
    
    if not (os.path.exists(train_csv_path) and os.path.exists(val_csv_path)):
        print(f"训练或验证数据不存在，准备数据...")
        return prepare_data()
    
    print(f"从{train_csv_path}和{val_csv_path}加载数据...")
    train_data = pd.read_csv(train_csv_path)
    val_data = pd.read_csv(val_csv_path)
    
    # 获取标签列表
    reports_df = pd.read_csv(os.path.join(DATA_PATH['base_dir'], DATA_PATH['reports_csv']))
    projections_df = pd.read_csv(os.path.join(DATA_PATH['base_dir'], DATA_PATH['projections_csv']))
    df = pd.merge(reports_df, projections_df, on='uid')
    all_labels = process_multiple_labels(df['Problems'])
    
    # 将字符串类型的labels列转为numpy数组
    train_labels = []
    for label_str in train_data['labels']:
        label_str = label_str.replace('[', '').replace(']', '')
        label_values = [float(x) for x in label_str.split()]
        train_labels.append(label_values)
    
    val_labels = []
    for label_str in val_data['labels']:
        label_str = label_str.replace('[', '').replace(']', '')
        label_values = [float(x) for x in label_str.split()]
        val_labels.append(label_values)
    
    return (
        {
            'image_path': train_data['image_path'].values,
            'labels': np.array(train_labels)
        },
        {
            'image_path': val_data['image_path'].values,
            'labels': np.array(val_labels)
        },
        all_labels
    )

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 准备数据
    prepare_data() 