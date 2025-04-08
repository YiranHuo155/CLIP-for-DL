"""
可视化模块，用于显示图像和模型结果
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix
from config import MODEL_CONFIG, LOG_CONFIG
import os

def display_image(image_np, title=None):
    """显示图像
    
    Args:
        image_np: 图像数组
        title: 图像标题
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt

def save_or_show_image(plt, filename=None):
    """保存或显示图像
    
    Args:
        plt: matplotlib图像对象
        filename: 保存的文件名
    """
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_training_history(train_losses, val_losses, filename=None):
    """绘制训练历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        filename: 保存的文件名
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    save_or_show_image(plt, filename)

def plot_confusion_matrix(y_true, y_pred, classes, filename=None):
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称
        filename: 保存的文件名
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    save_or_show_image(plt, filename)

def visualize_top_predictions(image, predictions, scores, filename=None):
    """可视化顶部预测结果
    
    Args:
        image: 图像数组
        predictions: 预测结果列表
        scores: 预测分数列表
        filename: 保存的文件名
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 显示图像
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Chest X-ray Image')
    
    # 显示预测结果
    y_pos = np.arange(len(predictions))
    ax2.barh(y_pos, scores, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictions)
    ax2.invert_yaxis()  # 标签从上到下
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def visualize_disease_distribution(disease_stats, filename=None):
    """可视化疾病分布
    
    Args:
        disease_stats: 疾病统计信息DataFrame
        filename: 保存的文件名
    """
    # 取频率前20的疾病
    top_diseases = disease_stats.sort_values('frequency', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='frequency', y='disease', data=top_diseases)
    ax.set_title('Top 20 Disease Frequency')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Disease')
    
    # 添加数值标签
    for i, v in enumerate(top_diseases['frequency']):
        ax.text(v + 0.1, i, str(int(v)), color='black', va='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def plot_metrics_comparison(metrics_dict, title='Performance Metrics', filename=None):
    """绘制评估指标比较
    
    Args:
        metrics_dict: 指标字典，键为指标名称，值为指标值
        title: 图表标题
        filename: 保存的文件名
    """
    # 筛选数值型指标
    numeric_metrics = {k: v for k, v in metrics_dict.items() 
                       if isinstance(v, (int, float)) and k != 'classification_report'}
    
    plt.figure(figsize=(10, 6))
    plt.bar(numeric_metrics.keys(), numeric_metrics.values())
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for i, (k, v) in enumerate(numeric_metrics.items()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def visualize_image_text_similarity(image_features, text_features, disease_list, filename=None):
    """可视化图像和文本的相似度
    
    Args:
        image_features: 图像特征张量
        text_features: 文本特征张量
        disease_list: 疾病列表
        filename: 保存的文件名
    """
    # 计算相似度矩阵
    similarity = (image_features @ text_features.T).detach().cpu().numpy()
    
    # 如果是批次，取第一个图像
    if len(similarity.shape) > 1 and similarity.shape[0] > 1:
        similarity = similarity[0]
    
    # 排序，获取前10个预测
    top_indices = np.argsort(similarity)[::-1][:10]
    top_diseases = [disease_list[i] for i in top_indices]
    top_scores = similarity[top_indices]
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=top_scores, y=top_diseases)
    ax.set_title('Top 10 Similar Diseases')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Disease')
    
    # 添加数值标签
    for i, v in enumerate(top_scores):
        ax.text(v + 0.01, i, f'{v:.4f}', color='black', va='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename) 