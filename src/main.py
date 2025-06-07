import os
import wandb
from data_loader import create_data_loaders
from model import CustomModel
from trainer import Trainer
from torch.utils.data import DataLoader
import torch
from utils import set_seed, setup_logging, ConfigManager
from visualization_utils import WandBVisualizer
from visualization_utils import VisualizationUtils

def main():
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = ConfigManager.load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config['reproducibility']['seed'])
    
    # Initialize wandb
    wandb.init(project=config['project']['name'], config=config)
    
    # Verify data directories exist
    train_dir = config['data']['train_data_dir']
    val_dir = config['data']['val_data_dir']
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")
    
    # Create all data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Initialize model
    model = CustomModel(
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    
    # Create model directory
    os.makedirs(config['model']['model_path'], exist_ok=True)
    
    # Initialize trainer with proper config access
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'epochs': config['training']['epochs'],
        'model_path': config['model']['model_path']
    }
    
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    
    # Train model
    trainer.train()
    
    # Add visualization after training
    visualizer = WandBVisualizer()
    
    # Plot training curves
    visualizer.plot_training_curves(
        run_id=wandb.run.id,
        metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'],
        save_path='training_curves.png'
    )
    
    # Add after training completion:
    vis_utils = VisualizationUtils(model=model, device=device)
    
    # Visualize feature distributions
    vis_utils.visualize_feature_distributions(
        dataloader=val_loader,
        save_path='feature_distributions.png'
    )
    
    # Visualize predictions on sample images
    vis_utils.visualize_predictions(
        image_path='path/to/sample/image.jpg',
        save_path='prediction_visualization.png'
    )
    
    # Visualize a batch of images with predictions
    images, labels = next(iter(val_loader))
    with torch.no_grad():
        predictions = model(images.to(device))
    vis_utils.visualize_batch(
        images=images,
        predictions=predictions.argmax(dim=1),
        save_path='batch_visualization.png'
    )
    
    wandb.finish()

if __name__ == '__main__':
    main() 