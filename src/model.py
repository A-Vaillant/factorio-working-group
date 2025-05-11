import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BinaryMatrixTransformCNN(nn.Module):
    filename = 'binary_model'
    
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
        
        
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x):
        # x shape: [batch, C, H, W]
        batch_size, C, H, W = x.size()
        
        # Project and reshape for attention
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, C/8]
        key = self.key(x).view(batch_size, -1, H * W)  # [B, C/8, HW]
        value = self.value(x).view(batch_size, -1, H * W)  # [B, C, HW]
        
        # Compute attention scores
        attention = F.softmax(torch.bmm(query, key), dim=-1)  # [B, HW, HW]
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x  # Residual connection
    
    
class AttentiveBinaryMatrixTransformCNN(BinaryMatrixTransformCNN):
    filename = 'attention_model'

    def __init__(self, channels=21, matrix_size=20):
        super(AttentiveBinaryMatrixTransformCNN, self).__init__()
        
        self.attention1 = SelfAttention(64)
        self.attention2 = SelfAttention(128)
        self.attention3 = SelfAttention(128)

        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        x = self.conv_out(x)
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