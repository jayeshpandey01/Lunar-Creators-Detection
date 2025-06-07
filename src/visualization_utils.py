import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import wandb
import seaborn as sns
import pandas as pd
from typing import List, Optional
from wandb.apis.public import Run

class VisualizationUtils:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def visualize_batch(self, images, predictions=None, save_path=None):
        """Visualize a batch of images with their predictions"""
        batch_size = images.size(0)
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        
        plt.figure(figsize=(15, 15))
        for idx in range(batch_size):
            plt.subplot(grid_size, grid_size, idx + 1)
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            if predictions is not None:
                plt.title(f'Pred: {predictions[idx]}')
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({"batch_visualization": wandb.Image(save_path)})
        plt.close()

    def generate_attention_maps(self, image_path):
        """Generate attention maps using Grad-CAM"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get the last convolutional layer
        target_layer = self.model.layer4[-1]
        
        # Register hooks
        feature_maps = []
        gradients = []
        
        def save_features(module, input, output):
            feature_maps.append(output)
            
        def save_grads(module, grad_in, grad_out):
            gradients.append(grad_out[0])
            
        handles = [
            target_layer.register_forward_hook(save_features),
            target_layer.register_full_backward_hook(save_grads)
        ]
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            score = output.max()
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Generate attention map
            features = feature_maps[0]
            grads = gradients[0]
            
            weights = grads.mean(dim=(2, 3), keepdim=True)
            attention_map = (weights * features).sum(dim=1, keepdim=True)
            attention_map = F.relu(attention_map)
            
            # Normalize and resize
            attention_map = F.interpolate(
                attention_map,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            
            attention_map = attention_map.squeeze().cpu().detach().numpy()
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            
            return attention_map
            
        finally:
            for handle in handles:
                handle.remove()

    def visualize_feature_distributions(self, dataloader, save_path=None):
        """Visualize feature distributions across the dataset"""
        features_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0].to(self.device)
                else:
                    images = batch.to(self.device)
                features = self.model.forward_features(images)
                features_list.append(features.cpu().numpy())
        
        features_array = np.concatenate(features_list, axis=0)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(features_array.mean(axis=(1, 2, 3)), bins=50)
        plt.title('Feature Mean Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(features_array.std(axis=(1, 2, 3)), bins=50)
        plt.title('Feature Std Distribution')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({"feature_distributions": wandb.Image(save_path)})
        plt.close()

    def visualize_predictions(self, image_path, save_path=None):
        """Visualize model predictions with attention maps"""
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions and attention map
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)
        attention_map = self.generate_attention_maps(image_path)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention map
        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(attention_map, alpha=0.5, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Predictions
        plt.subplot(1, 3, 3)
        probs = F.softmax(predictions, dim=1)[0]
        top_probs, top_indices = probs.topk(5)
        plt.bar(range(5), top_probs.cpu().numpy())
        plt.title('Top 5 Predictions')
        
        if save_path:
            plt.savefig(save_path)
            wandb.log({
                "predictions_visualization": wandb.Image(save_path),
                "top_probability": top_probs[0].item()
            })
        plt.close() 

class WandBVisualizer:
    """Utility class for visualizing WandB results"""
    
    @staticmethod
    def get_project_runs(project_name: str, entity: str, limit: int = 5):
        """Fetch most recent runs from a project"""
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project_name}", per_page=limit)
        return runs
    
    @staticmethod
    def compare_recent_runs(project_name: str, 
                          entity: str,
                          metric: str,
                          limit: int = 5,
                          smoothing: float = 0.9,
                          save_path: Optional[str] = None):
        """Compare most recent runs in a project"""
        plt.figure(figsize=(12, 6))
        
        runs = WandBVisualizer.get_project_runs(project_name, entity, limit)
        
        for run in runs:
            history = run.scan_history()
            df = pd.DataFrame(history)
            
            if metric in df.columns:
                plt.plot(df[metric].ewm(alpha=(1 - smoothing)).mean(), 
                        label=f'Run: {run.name}')
        
        plt.title(f'Comparison of {metric} Across Recent Runs')
        plt.xlabel('Steps')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 