"""
Utility Functions for the Project
Purpose:
1. Common helper functions used across different modules
2. Metric calculations and monitoring
3. System and environment setup
4. Data preprocessing helpers
5. Visualization tools
6. Model checkpointing and loading
7. Configuration management
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import logging
from datetime import datetime
import wandb

class ConfigManager:
    """Handles loading and saving configuration files"""
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_config(config, save_path):
        with open(save_path, 'w') as file:
            yaml.dump(config, file)

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MetricTracker:
    """Track and compute various training metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'learning_rate': []
        }
    
    def update(self, metric_dict):
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, metric):
        return np.mean(self.metrics[metric])

class Visualizer:
    """Visualization utilities for training and results"""
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training metrics history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
        plt.close()

class ModelCheckpoint:
    """Handle model checkpointing"""
    def __init__(self, save_dir, metric_name='val_loss', mode='min'):
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, metric_value, filename=None):
        """Save model checkpoint if metric improves"""
        improved = (self.mode == 'min' and metric_value < self.best_value) or \
                  (self.mode == 'max' and metric_value > self.best_value)
        
        if improved:
            self.best_value = metric_value
            if filename is None:
                filename = f'best_model_{self.metric_name}_{metric_value:.4f}.pth'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'best_{self.metric_name}': self.best_value
            }
            
            save_path = os.path.join(self.save_dir, filename)
            torch.save(checkpoint, save_path)
            logging.info(f'Saved checkpoint: {save_path}')
            return True
        return False

def calculate_metrics(outputs, targets):
    """Calculate various performance metrics"""
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_model_architecture(model, save_path):
    """Save model architecture summary"""
    with open(save_path, 'w') as f:
        f.write(str(model))

def memory_usage_summary():
    """Get GPU memory usage summary"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'cached': torch.cuda.memory_reserved() / 1024**2
        }
    return None 