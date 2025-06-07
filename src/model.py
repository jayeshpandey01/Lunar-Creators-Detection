import torch
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(CustomModel, self).__init__()
        
        # Use EfficientNet as base model
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify the classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Add additional layers for better feature extraction
        self.model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
    def forward(self, x):
        return self.model(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss 