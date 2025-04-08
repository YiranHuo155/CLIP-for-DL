"""
Configuration file for CLIP chest X-ray project
"""

import torch
import os

# Data paths
DATA_PATH = {
    'base_dir': 'data',
    'image_dir': 'data/images_normalized',
    'reports_csv': 'indiana_reports.csv',
    'projections_csv': 'indiana_projections.csv',
    'train_data': 'train_data.csv',
    'val_data': 'val_data.csv'
}

# Model parameters
MODEL_CONFIG = {
    'image_size': 224,
    'patch_size': 16,
    'image_embedding_size': 2048,
    'text_embedding_size': 768,
    'shared_embedding_size': 512,
    'temperature': 0.07,
    'dropout_rate': 0.1,
    'max_text_length': 200,
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT'
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 1, # 这里为了测试，设置为1；后续可调整为100
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'warmup_steps': 1000,
    'validation_interval': 1,
    'early_stopping_patience': 5,
    'scheduler_factor': 0.1,
    'scheduler_patience': 2,
    'num_workers': 4
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging configuration
LOG_CONFIG = {
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'log_interval': 100,
    'save_top_k': 3
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_degrees': 10,
    'translate': (0.1, 0.1),
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'random_horizontal_flip_p': 0.5,
    'random_rotation_degrees': 10,
    'random_affine_translate': (0.1, 0.1)
}

# Prediction parameters
PREDICTION_CONFIG = {
    'threshold': 0.5,
    'top_k': 3,
    'min_confidence': 0.3
}

# Create necessary directories
for directory in [DATA_PATH['base_dir'], DATA_PATH['image_dir'], 
                 LOG_CONFIG['log_dir'], LOG_CONFIG['checkpoint_dir']]:
    os.makedirs(directory, exist_ok=True) 