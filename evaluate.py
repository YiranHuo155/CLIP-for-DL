import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import load_data, ChestXrayDataset, get_data_transforms
from disease_analysis import analyze_disease_distribution, get_disease_cooccurrence
from config import DATA_PATH, MODEL_CONFIG, DEVICE
import logging
import os

def evaluate_original_method():
    """评估原始方法（只使用第一个诊断）的性能"""
    df = pd.read_csv(DATA_PATH['reports'])
    
    # 只取第一个诊断
    df['first_diagnosis'] = df['Problems'].apply(
        lambda x: x.split(',')[0] if isinstance(x, str) else x
    )
    
    # 统计诊断分布
    diagnosis_counts = df['first_diagnosis'].value_counts()
    rare_diseases = diagnosis_counts[diagnosis_counts == 1].index.tolist()
    
    print("\n=== 原始方法统计 ===")
    print(f"总样本数: {len(df)}")
    print(f"唯一诊断数: {len(diagnosis_counts)}")
    print(f"只出现一次的诊断数: {len(rare_diseases)}")
    print(f"只出现一次的诊断占比: {len(rare_diseases)/len(diagnosis_counts):.2%}")
    
    return {
        'total_samples': len(df),
        'unique_diagnoses': len(diagnosis_counts),
        'rare_diseases': len(rare_diseases),
        'rare_disease_ratio': len(rare_diseases)/len(diagnosis_counts)
    }

def evaluate_multilabel_method():
    """评估多标签方法的性能"""
    df = pd.read_csv(DATA_PATH['reports'])
    
    # 处理所有诊断
    all_diagnoses = []
    for problems in df['Problems']:
        if isinstance(problems, str):
            diagnoses = [d.strip() for d in problems.split(',')]
            all_diagnoses.extend(diagnoses)
    
    # 统计诊断分布
    diagnosis_counts = pd.Series(all_diagnoses).value_counts()
    rare_diseases = diagnosis_counts[diagnosis_counts == 1].index.tolist()
    
    print("\n=== 多标签方法统计 ===")
    print(f"总样本数: {len(df)}")
    print(f"唯一诊断数: {len(diagnosis_counts)}")
    print(f"只出现一次的诊断数: {len(rare_diseases)}")
    print(f"只出现一次的诊断占比: {len(rare_diseases)/len(diagnosis_counts):.2%}")
    
    # 分析疾病共现
    cooccurrence = get_disease_cooccurrence(df)
    avg_cooccurrence = cooccurrence.mean().mean()
    print(f"平均疾病共现次数: {avg_cooccurrence:.2f}")
    
    return {
        'total_samples': len(df),
        'unique_diagnoses': len(diagnosis_counts),
        'rare_diseases': len(rare_diseases),
        'rare_disease_ratio': len(rare_diseases)/len(diagnosis_counts),
        'avg_cooccurrence': avg_cooccurrence
    }

def compare_prediction_results():
    """比较两种方法的预测结果"""
    # 加载最新的预测结果
    if os.path.exists('predictions_multilabel.pt') and os.path.exists('predictions_original.pt'):
        multilabel_results = torch.load('predictions_multilabel.pt')
        original_results = torch.load('predictions_original.pt')
        
        print("\n=== 预测性能比较 ===")
        print("原始方法:")
        print(f"准确率: {original_results['accuracy']:.4f}")
        print(f"宏平均F1分数: {original_results['macro_f1']:.4f}")
        print(f"微平均F1分数: {original_results['micro_f1']:.4f}")
        
        print("\n多标签方法:")
        print(f"准确率: {multilabel_results['accuracy']:.4f}")
        print(f"宏平均F1分数: {multilabel_results['macro_f1']:.4f}")
        print(f"微平均F1分数: {multilabel_results['micro_f1']:.4f}")
        
        return {
            'original': original_results,
            'multilabel': multilabel_results
        }
    else:
        print("未找到预测结果文件，请先运行训练和预测")
        return None

def plot_comparison(original_stats, multilabel_stats):
    """绘制比较图表"""
    # 创建图表目录
    os.makedirs('evaluation_plots', exist_ok=True)
    
    # 1. 稀有疾病比例对比
    plt.figure(figsize=(10, 6))
    methods = ['Original', 'Multilabel']
    rare_ratios = [original_stats['rare_disease_ratio'], 
                   multilabel_stats['rare_disease_ratio']]
    plt.bar(methods, rare_ratios)
    plt.title('Rare Disease Ratio Comparison')
    plt.ylabel('Ratio')
    plt.savefig('evaluation_plots/rare_disease_comparison.png')
    plt.close()
    
    # 2. 诊断数量对比
    plt.figure(figsize=(10, 6))
    metrics = ['Total Samples', 'Unique Diagnoses', 'Rare Diseases']
    original_values = [original_stats['total_samples'],
                      original_stats['unique_diagnoses'],
                      original_stats['rare_diseases']]
    multilabel_values = [multilabel_stats['total_samples'],
                        multilabel_stats['unique_diagnoses'],
                        multilabel_stats['rare_diseases']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original')
    plt.bar(x + width/2, multilabel_values, width, label='Multilabel')
    plt.xlabel('Metrics')
    plt.ylabel('Count')
    plt.title('Diagnosis Statistics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('evaluation_plots/diagnosis_statistics.png')
    plt.close()

def main():
    """主函数"""
    # 评估原始方法
    original_stats = evaluate_original_method()
    
    # 评估多标签方法
    multilabel_stats = evaluate_multilabel_method()
    
    # 比较预测结果
    prediction_comparison = compare_prediction_results()
    
    # 绘制比较图表
    plot_comparison(original_stats, multilabel_stats)
    
    # 保存评估结果
    results = {
        'original': original_stats,
        'multilabel': multilabel_stats,
        'prediction_comparison': prediction_comparison
    }
    
    pd.DataFrame(results).to_csv('evaluation_results.csv')
    print("\n评估结果已保存到 evaluation_results.csv")
    print("比较图表已保存到 evaluation_plots 目录")

if __name__ == '__main__':
    main() 