import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import os
from utils import MetricTracker, ModelCheckpoint, calculate_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader)
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        self.metric_tracker = MetricTracker()
        self.checkpoint = ModelCheckpoint(save_dir='checkpoints')
        
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}') as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                data, target = batch_data[0], batch_data[1]
                data, target = data.to(self.device), target.to(self.device)
                
                # Mixed precision training
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
                
                metrics = calculate_metrics(output, target)
                self.metric_tracker.update(metrics)
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        
        # Log to wandb
        wandb.log({
            'val_loss': val_loss,
            'val_accuracy': accuracy
        })
        
        return val_loss, accuracy
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                         os.path.join(self.config['model_path'], 'best_model.pth'))
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}') 