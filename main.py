import torch
import pandas as pd
from disease_analysis import (
    analyze_disease_distribution,
    create_rich_prompts,
    predict_zero_shot,
    evaluate_predictions
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

def load_models(device):
    """加载所有需要的模型"""
    from transformers import AutoTokenizer, AutoModel
    import torchvision.models as models
    import torch.nn as nn
    
    # 加载ResNet模型
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model = resnet_model.to(device)
    
    # 加载Bio_ClinicalBERT
    model_name = 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name).to(device)
    
    # 创建图像和文本投影网络
    shared_embedding_size = 512
    image_embedding_size = 2048
    text_embedding_size = 768
    
    class ImageProjection(nn.Module):
        def __init__(self, image_embedding_size, shared_embedding_size):
            super(ImageProjection, self).__init__()
            self.image_projection = nn.Linear(image_embedding_size, shared_embedding_size)
            self.gelu = nn.GELU()
            self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
            self.dropout = nn.Dropout(0.1)
            self.layer_norm = nn.LayerNorm(shared_embedding_size)

        def forward(self, image_embeddings):
            projected_embeddings = self.image_projection(image_embeddings)
            
            x = self.gelu(projected_embeddings)
            x = self.fc(x)
            x = self.dropout(x)
            x = x + projected_embeddings
            x = self.layer_norm(x)
            
            return x
            
    class TextProjection(nn.Module):
        def __init__(self, text_embedding_size, shared_embedding_size):
            super(TextProjection, self).__init__()
            self.text_projection = nn.Linear(text_embedding_size, shared_embedding_size)
            self.gelu = nn.GELU()
            self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
            self.dropout = nn.Dropout(0.1)
            self.layer_norm = nn.LayerNorm(shared_embedding_size)

        def forward(self, text_embeddings):
            projected_embeddings = self.text_projection(text_embeddings)
            
            x = self.gelu(projected_embeddings)
            x = self.fc(x)
            x = self.dropout(x)
            x = x + projected_embeddings
            x = self.layer_norm(x)
            
            return x
    
    image_projector = ImageProjection(image_embedding_size, shared_embedding_size).to(device)
    text_projector = TextProjection(text_embedding_size, shared_embedding_size).to(device)
    
    return {
        'resnet': resnet_model,
        'text_model': text_model,
        'tokenizer': tokenizer,
        'image_projector': image_projector,
        'text_projector': text_projector
    }

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    df_reports = pd.read_csv('indiana_reports.csv')
    
    # 分析疾病分布
    print("Analyzing disease distribution...")
    disease_stats = analyze_disease_distribution(df_reports)
    print("\nTop 10 most common diseases:")
    print(disease_stats.head(10))
    
    # 创建丰富的提示模板
    print("\nCreating rich prompts...")
    prompts = create_rich_prompts(disease_stats)
    
    # 加载模型
    print("\nLoading models...")
    models = load_models(device)
    
    # 准备数据加载器
    test_dataset = ImageTextDataset(test_captions.image_path.values, 
                                  test_captions.label.values,
                                  image_size=224)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 进行预测和评估
    print("\nPerforming zero-shot predictions...")
    all_predictions = []
    all_labels = []
    
    for images, labels in tqdm(test_dataloader):
        for image, label in zip(images, labels):
            # 使用新的 predict_zero_shot 函数，而不是旧的 zero_shot_predict
            predictions = predict_zero_shot(
                images=image,
                models=models,
                disease_list=list(disease_stats.index),
                top_k=3,
                prompts=prompts,
                use_enhanced_prompts=True
            )
            
            # predictions 现在是一个字典列表
            all_predictions.append(predictions[0]['disease'])  # 取top1预测
            all_labels.append(label.split(';')[0])  # 取第一个标签
            
    # 计算总体指标
    print("\nCalculating metrics...")
    print(classification_report(all_labels, all_predictions))

if __name__ == "__main__":
    main() 