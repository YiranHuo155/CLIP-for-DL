import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import MODEL_CONFIG, TRAINING_CONFIG, LOG_CONFIG, DEVICE, DATA_PATH
from prepare_data import load_data, get_data_transforms, ChestXrayDataset, preprocess_image
from disease_analysis import analyze_disease_distribution, get_disease_cooccurrence, create_rich_prompts, predict_zero_shot, evaluate_predictions
from train import train_clip
from visualization import plot_training_history, visualize_disease_distribution, plot_metrics_comparison, visualize_top_predictions

def main():
    """主函数"""
    # 设置日志
    os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(LOG_CONFIG['checkpoint_dir'], exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(LOG_CONFIG['log_dir'], 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据
    logging.info("Loading data...")
    train_data, val_data, all_labels = load_data()
    
    # 分析疾病分布
    logging.info("Analyzing disease distribution...")
    disease_stats = analyze_disease_distribution(pd.read_csv(os.path.join(DATA_PATH['base_dir'], DATA_PATH['reports_csv'])))
    disease_list = disease_stats.index.tolist()
    
    # 可视化疾病分布
    visualize_disease_distribution(
        disease_stats.reset_index().rename(columns={'index': 'disease'}),
        os.path.join(LOG_CONFIG['log_dir'], 'disease_distribution.png')
    )
    
    # 分析疾病共现关系
    logging.info("Analyzing disease co-occurrence...")
    cooccurrence = get_disease_cooccurrence(pd.read_csv(os.path.join(DATA_PATH['base_dir'], DATA_PATH['reports_csv'])))
    
    # 创建丰富的提示模板
    logging.info("Creating prompt templates...")
    prompts = create_rich_prompts(disease_stats)
    
    # 设置数据转换
    train_transform, val_transform = get_data_transforms()
    
    # 创建数据集
    train_dataset = ChestXrayDataset(
        train_data['image_path'],
        train_data['labels'],
        MODEL_CONFIG['image_size'],
        train_transform
    )
    
    val_dataset = ChestXrayDataset(
        val_data['image_path'],
        val_data['labels'],
        MODEL_CONFIG['image_size'],
        val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # 训练模型
    logging.info("Starting model training...")
    models, train_history = train_clip(train_loader, val_loader, disease_list)
    logging.info("Model training completed")
    
    # 绘制训练历史
    plot_training_history(
        train_history['train_loss'],
        train_history['val_loss'],
        os.path.join(LOG_CONFIG['log_dir'], 'training_history.png')
    )
    
    # 零样本预测和评估
    logging.info("Performing zero-shot prediction...")
    all_predictions = []
    all_scores = []
    all_images = []
    all_true_labels = []
    
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(DEVICE)
        batch_predictions, batch_scores = predict_zero_shot(
            images, 
            models, 
            disease_list, 
            top_k=3,
            prompts=prompts,
            use_enhanced_prompts=True
        )
        all_predictions.extend(batch_predictions)
        all_scores.extend(batch_scores)
        
        # 保存一些图像用于可视化
        if batch_idx < 5:  # 只保存前5个批次的图像
            all_images.extend(images.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    # 评估预测结果
    logging.info("Evaluating predictions...")
    
    # 提取验证集的实际标签
    true_labels = []
    for _, labels in val_loader:
        true_labels.extend(labels.cpu().numpy())
    
    metrics = evaluate_predictions(
        all_predictions,
        true_labels,
        disease_list
    )
    
    # 可视化评估指标
    plot_metrics_comparison(
        metrics,
        'Zero-shot Prediction Performance',
        os.path.join(LOG_CONFIG['log_dir'], 'metrics_comparison.png')
    )
    
    # 可视化一些预测结果
    for i in range(min(5, len(all_images))):
        image = all_images[i].transpose(1, 2, 0)  # 转换为HWC格式
        preds = all_predictions[i]
        scores = all_scores[i]
        
        visualize_top_predictions(
            image,
            preds,
            scores,
            os.path.join(LOG_CONFIG['log_dir'], f'prediction_sample_{i}.png')
        )
    
    # 保存评估结果
    logging.info("Saving evaluation results...")
    results = {
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'micro_f1': metrics['micro_f1'],
        'weighted_f1': metrics['weighted_f1']
    }
    
    pd.DataFrame([results]).to_csv(
        os.path.join(LOG_CONFIG['log_dir'], 'evaluation_metrics.csv'),
        index=False
    )
    
    pd.DataFrame(metrics['classification_report']).to_csv(
        os.path.join(LOG_CONFIG['log_dir'], 'classification_report.csv'),
        index=False
    )
    
    logging.info("All done!")

if __name__ == "__main__":
    main() 