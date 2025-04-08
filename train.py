import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import time
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.models as models
import os
import logging
from config import MODEL_CONFIG, TRAINING_CONFIG, LOG_CONFIG, DEVICE
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report
from disease_analysis import predict_zero_shot, evaluate_predictions

class ImageTextDataset(Dataset):
    def __init__(self, image_filenames, text_list, image_size=224):
        self.image_filenames = image_filenames
        self.text_list = text_list
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join("data", "images_normalized", self.image_filenames[idx])
        image = self.preprocess_image(image_path)
        image = self.transform(image)
        text = self.text_list[idx]
        return image, text
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

class ImageProjection(nn.Module):
    """图像投影模块"""
    
    def __init__(self, image_embedding_size, shared_embedding_size):
        super(ImageProjection, self).__init__()
        self.image_projection = nn.Linear(image_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        self.layer_norm = nn.LayerNorm(shared_embedding_size)

    def forward(self, image_embeddings):
        projected = self.image_projection(image_embeddings)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextProjection(nn.Module):
    """文本投影模块"""
    
    def __init__(self, text_embedding_size, shared_embedding_size):
        super(TextProjection, self).__init__()
        self.text_projection = nn.Linear(text_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        self.layer_norm = nn.LayerNorm(shared_embedding_size)

    def forward(self, text_embeddings):
        projected = self.text_projection(text_embeddings)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    """计算交叉熵损失"""
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_clip_loss_function(text_projection, image_projection, temperature=MODEL_CONFIG['temperature'], mode="eval"):
    """计算CLIP对比损失
    
    Args:
        text_projection: 文本投影特征
        image_projection: 图像投影特征
        temperature: 温度参数
        mode: 'train' 或 'eval'
        
    Returns:
        train模式下返回损失值，eval模式下返回相似度矩阵
    """
    logits = (text_projection @ image_projection.T) / temperature
    if mode == "train":
        images_similarity = image_projection @ image_projection.T
        texts_similarity = text_projection @ text_projection.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()
    elif mode == "eval":
        return logits
    else:
        logging.error("Invalid mode for contrastive loss")
        return None

def contrastive_loss(image_features, text_features, temperature=1.0):
    """计算对比损失
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        temperature: 温度参数
        
    Returns:
        损失值
    """
    # 计算相似度矩阵
    logits = (image_features @ text_features.T) / temperature
    
    # 创建标签（对角线为1，其他为0）
    labels = torch.arange(len(image_features)).to(DEVICE)
    
    # 计算交叉熵损失
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2.0

def multilabel_contrastive_loss(image_features, text_features, labels, temperature=1.0):
    """计算多标签对比损失
    
    Args:
        image_features: 图像特征 [batch_size, embedding_dim]
        text_features: 文本特征 [num_classes, embedding_dim]
        labels: 多标签标签 [batch_size, num_classes]
        temperature: 温度参数
        
    Returns:
        损失值
    """
    # 规范化特征
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # 计算相似度，值域为[-1/temp, 1/temp]
    similarities = (image_features @ text_features.T) / temperature
    
    # 记录similarities的统计信息，帮助调试
    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
        logging.error(f"相似度计算出现NaN或Inf: {similarities}")
    
    batch_size = labels.size(0)
    num_classes = text_features.size(0)
    
    # 确保标签格式正确
    if labels.dim() != 2 or labels.size(1) != num_classes:
        logging.error(f"标签格式不正确: shape={labels.shape}, 期望shape=[{batch_size}, {num_classes}]")
        # 尝试修复标签格式
        if labels.dim() == 1:
            labels = F.one_hot(labels.long(), num_classes).float()
        else:
            # 尝试使用最基本的对比损失替代
            logging.warning("使用基本对比损失作为替代")
            return contrastive_loss(image_features, text_features, temperature)
    
    # 计算正样本和负样本的损失，使用clip防止数值溢出
    similarities_clipped = torch.clamp(similarities, -50.0, 50.0)
    pos_probs = torch.sigmoid(similarities_clipped)
    neg_probs = 1 - pos_probs
    
    # 计算损失
    pos_loss = -torch.sum(torch.log(pos_probs + 1e-8) * labels) / (torch.sum(labels) + 1e-8)
    neg_loss = -torch.sum(torch.log(neg_probs + 1e-8) * (1 - labels)) / (torch.sum(1 - labels) + 1e-8)
    
    loss = (pos_loss + neg_loss) / 2.0
    
    # 检查损失值是否合理
    if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
        logging.error(f"多标签对比损失异常: {loss}")
        logging.error(f"正样本损失: {pos_loss}, 负样本损失: {neg_loss}")
        # 回退到基本对比损失
        return contrastive_loss(image_features, text_features, temperature)
    
    return loss

def train_epoch(models, train_loader, optimizer, epoch):
    """训练一个epoch
    
    Args:
        models: 模型字典
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前epoch
        
    Returns:
        平均损失
    """
    # 设置训练模式，但跳过tokenizer
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'train'):
            model.train()
    
    total_loss = 0
    num_classes = models['text_features'].size(0)
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            
            # 检查标签格式
            if isinstance(labels, tuple):
                logging.warning(f"标签是元组，尝试使用第一个元素: {type(labels)}")
                labels = labels[0]
            
            if not isinstance(labels, torch.Tensor):
                logging.warning(f"标签不是tensor，尝试转换: {type(labels)}")
                labels = torch.tensor(labels, dtype=torch.float32)
            
            # 确保标签是正确的形状: [batch_size, num_classes]
            if labels.dim() == 1:
                logging.warning("标签是一维的，尝试转换为one-hot编码")
                labels = F.one_hot(labels.long(), num_classes).float()
            elif labels.dim() > 2:
                logging.warning(f"标签维度过高: {labels.shape}，尝试转换")
                labels = labels.view(labels.size(0), -1)
                
            # 如果标签维度与类别数不匹配，打印警告并尝试截断或扩展
            if labels.size(1) != num_classes:
                logging.warning(f"标签维度不匹配: 标签大小={labels.shape}，类别数={num_classes}")
                if labels.size(1) > num_classes:
                    labels = labels[:, :num_classes]
                else:
                    padding = torch.zeros(labels.size(0), num_classes - labels.size(1), device=labels.device)
                    labels = torch.cat([labels, padding], dim=1)
            
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            try:
                # 获取图像特征
                image_embeddings = models['resnet'](images)
                image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
                image_features = models['image_projector'](image_embeddings)
                
                # 获取文本特征
                text_features = models['text_features']  # 预计算的文本特征
                
                # 计算损失
                loss = multilabel_contrastive_loss(
                    image_features, 
                    text_features, 
                    labels,
                    MODEL_CONFIG['temperature']
                )
                
                # 如果损失太大或是NaN，跳过这个批次
                if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
                    logging.error(f"批次 {batch_idx} 损失异常: {loss}，跳过该批次")
                    continue
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % LOG_CONFIG['log_interval'] == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    logging.info(
                        f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {avg_loss:.4f}'
                    )
            except Exception as e:
                logging.error(f"批次 {batch_idx} 训练出错: {str(e)}")
                continue
    
    return total_loss / len(train_loader)

def validate(models, val_loader):
    """验证模型
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        
    Returns:
        验证损失
    """
    # 设置评估模式，但跳过tokenizer
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'eval'):
            model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 获取图像特征
            image_embeddings = models['resnet'](images)
            image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
            image_features = models['image_projector'](image_embeddings)
            
            # 获取文本特征
            text_features = models['text_features']
            
            # 计算损失
            loss = multilabel_contrastive_loss(
                image_features,
                text_features,
                labels,
                MODEL_CONFIG['temperature']
            )
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def save_checkpoint(state, is_best, checkpoint_dir):
    """保存模型检查点
    
    Args:
        state: 模型状态字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
    """
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)

def load_checkpoint(models, optimizer, filename):
    """加载检查点
    
    Args:
        models: 模型字典
        optimizer: 优化器
        filename: 加载文件名
        
    Returns:
        开始epoch和最佳损失
    """
    if not os.path.exists(filename):
        return 0, float('inf')
    
    checkpoint = torch.load(filename)
    
    for name, model in models.items():
        if name in checkpoint['models']:
            model.load_state_dict(checkpoint['models'][name])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    logging.info(f'Loaded checkpoint from {filename}')
    return checkpoint['epoch'], checkpoint['loss']

def predict_and_evaluate(models, val_loader, disease_list):
    """进行预测并评估
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        disease_list: 疾病列表
        
    Returns:
        predictions: 预测结果
        labels: 真实标签
    """
    # 设置评估模式
    for model in models.values():
        model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 获取图像特征
            image_embeddings = models['resnet'](images)
            image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
            image_features = models['image_projector'](image_embeddings)
            
            # 获取文本特征
            text_features = models['text_features']
            
            # 计算预测结果
            predictions = predict_multilabel(
                image_features,
                text_features,
                threshold=0.5
            )
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # 合并所有批次的结果
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return predictions, labels

class AverageMeter:
    """跟踪指标的平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_clip(train_loader, val_loader, disease_list):
    """训练CLIP模型
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        disease_list: 疾病列表
        
    Returns:
        models: 训练好的模型
        train_history: 训练历史
    """
    # 初始化模型
    models = initialize_models(DEVICE)
    
    # 预计算文本特征
    try:
        text_features = get_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        )
        models['text_features'] = text_features
    except Exception as e:
        logging.error(f"计算文本特征时出错: {e}")
        raise
    
    # 设置优化器
    params = []
    for name, model in models.items():
        if name not in ['tokenizer', 'text_features'] and hasattr(model, 'parameters'):
            params.extend(list(model.parameters()))
    
    optimizer = torch.optim.AdamW(
        params,
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG['epochs'],
        eta_min=TRAINING_CONFIG['min_learning_rate']
    )
    
    # 训练历史
    train_history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 尝试加载检查点
    start_epoch, best_val_loss = load_checkpoint(
        models,
        optimizer,
        os.path.join(LOG_CONFIG['checkpoint_dir'], 'model_best.pth')
    )
    
    # 训练循环
    for epoch in range(start_epoch, TRAINING_CONFIG['epochs']):
        # 训练一个epoch
        train_loss = train_epoch(models, train_loader, optimizer, epoch)
        train_history['train_loss'].append(train_loss)
        
        # 验证
        val_loss = validate(models, val_loader)
        train_history['val_loss'].append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练信息
        logging.info(
            f'Epoch {epoch} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # 评估当前模型性能
        logging.info("执行当前epoch评估...")
        # 进行零样本预测
        all_predictions = []
        all_true_labels = []
        
        try:
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    try:
                        batch_predictions, _ = predict_zero_shot(
                            images, models, disease_list, top_k=3
                        )
                        all_predictions.extend(batch_predictions)
                        all_true_labels.extend(labels.cpu().numpy())
                    except Exception as e:
                        logging.error(f"预测时出错: {e}")
                        continue
            
            # 计算评估指标
            try:
                metrics = evaluate_predictions(all_predictions, all_true_labels, disease_list)
                logging.info(
                    f'Epoch {epoch} 评估结果 - '
                    f'Accuracy: {metrics["accuracy"]:.4f}, '
                    f'Macro F1: {metrics["macro_f1"]:.4f}, '
                    f'Micro F1: {metrics["micro_f1"]:.4f}'
                )
            except Exception as e:
                logging.error(f"计算评估指标时出错: {e}")
        except Exception as e:
            logging.error(f"评估过程中出错: {e}")
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 只为可序列化的模型创建state_dict
        model_states = {}
        for name, model in models.items():
            if name != 'tokenizer' and name != 'text_features' and hasattr(model, 'state_dict'):
                model_states[name] = model.state_dict()
        
        state = {
            'epoch': epoch + 1,
            'models': model_states,
            'optimizer': optimizer.state_dict(),
            'loss': val_loss
        }
        
        save_checkpoint(state, is_best, LOG_CONFIG['checkpoint_dir'])
        
        # 早停检查
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            logging.info(f'早停: {patience_counter} epochs没有改善')
            break
    
    # 加载最佳模型
    load_checkpoint(
        models,
        optimizer,
        os.path.join(LOG_CONFIG['checkpoint_dir'], 'model_best.pth')
    )
    
    return models, train_history

def predict_multilabel(image_features, text_features, threshold=0.5):
    """多标签预测
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        threshold: 预测阈值
        
    Returns:
        torch.Tensor: 预测结果
    """
    # 计算相似度
    similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
    # 使用sigmoid将相似度转换为概率
    probabilities = torch.sigmoid(similarities)
    # 根据阈值进行预测
    predictions = (probabilities > threshold).float()
    return predictions

def initialize_models(device):
    """初始化所有模型
    
    Args:
        device: 计算设备
        
    Returns:
        dict: 包含所有模型的字典
    """
    # 加载ResNet模型
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model = resnet_model.to(device)
    
    # 创建图像投影层
    image_projector = ImageProjection(
        MODEL_CONFIG['image_embedding_size'], 
        MODEL_CONFIG['shared_embedding_size']
    ).to(device)
    
    # 加载文本模型
    model_name = MODEL_CONFIG['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 直接加载模型，已降级transformers版本解决init_empty_weights问题
    text_model = AutoModel.from_pretrained(model_name).to(device)
    
    # 创建文本投影层
    text_projector = TextProjection(
        MODEL_CONFIG['text_embedding_size'], 
        MODEL_CONFIG['shared_embedding_size']
    ).to(device)
    
    return {
        'resnet': resnet_model,
        'image_projector': image_projector,
        'text_model': text_model,
        'text_projector': text_projector,
        'tokenizer': tokenizer
    }

def calculate_multilabel_metrics(predictions, labels, disease_list):
    """计算多标签预测的评估指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        disease_list: 疾病列表
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 将张量转换为numpy数组
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # 计算各种指标
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    # 生成分类报告
    report = classification_report(
        labels,
        predictions,
        target_names=disease_list,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report
    }

def get_text_features(diseases, tokenizer, text_model, text_projector, device=DEVICE):
    """获取疾病文本的特征表示
    
    Args:
        diseases: 疾病列表
        tokenizer: 分词器
        text_model: 文本编码器
        text_projector: 文本投影层
        device: 计算设备
        
    Returns:
        torch.Tensor: 文本特征
    """
    prompts = [f"This is a chest X-ray showing {disease}." for disease in diseases]
    
    # 对文本进行编码
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    ).to(device)
    
    # 获取文本特征
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        text_features = text_projector(text_embeddings)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features 