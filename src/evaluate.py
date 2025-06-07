import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from visualization_utils import WandBVisualizer

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels.numpy())
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        all_labels, 
        all_preds, 
        normalize='true'
    )
    
    # Visualize confusion matrix
    visualizer = WandBVisualizer()
    visualizer.plot_confusion_matrix(
        confusion_matrix=conf_matrix,
        class_names=class_names,
        save_path='confusion_matrix.png'
    )
    
    return conf_matrix 