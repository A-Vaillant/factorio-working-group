import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BinaryMatrixTransformCNN(nn.Module):
    def __init__(self, channels=21, matrix_size=20):
        super(BinaryMatrixTransformCNN, self).__init__()
        
        # Multiple conv layers with padding to maintain spatial dimensions
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Project back to original dimensions
        self.conv_out = nn.Conv2d(128, channels, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [batch, 21, 20, 20]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        # Output shape: [batch, 21, 20, 20]
        return x

    def predict(self, X):
        """Make a prediction with thresholding"""
        self.eval()
        with torch.no_grad():
            # Get raw outputs
            raw_outputs = self(X)
            
            # Apply thresholding
            thresholded_outputs = (raw_outputs > 0.5).float()
            
            return thresholded_outputs