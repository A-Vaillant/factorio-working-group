import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQCNN(nn.Module):
    def __init__(self, input_channels=7, output_channels=7):
        super(DeepQCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        
        # MaxPooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # First dense layer
        self.dense1 = nn.Linear(64 * 10 * 10 + (1 if True else 0), 512)  # +1 for condition if not None
        
        # Second dense layer
        self.dense2 = nn.Linear(512, output_channels * 20 * 20)
        
    def forward(self, level, condition=None):
        # Level input processing: level shape should be [batch_size, 7, 20, 20]
        x = F.relu(self.conv1(level))
        x = self.maxpool(x)  # Shape: [batch_size, 32, 10, 10]
        
        x = F.relu(self.conv2(x))  # Shape: [batch_size, 64, 10, 10]
        x = F.relu(self.conv3(x))  # Shape: [batch_size, 64, 10, 10]
        
        # Flatten
        x = x.view(x.size(0), -1)  # Shape: [batch_size, 64 * 10 * 10]
        
        # Concatenate condition if provided
        if condition is not None:
            # Ensure condition is properly shaped [batch_size, 1]
            if len(condition.shape) == 1:
                condition = condition.unsqueeze(1)
            x = torch.cat((x, condition), dim=1)
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        
        # Reshape to output format [batch_size, 7, 20, 20]
        action = x.view(x.size(0), 7, 20, 20)
        
        return action

class MatrixClassifier(nn.Module):
    def __init__(self, input_channels=7):
        super(MatrixClassifier, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        
        # MaxPooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Flattened input size after convolutions and pooling
        self.flatten_size = 128 * 5 * 5
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: 0 or 1
        
    def forward(self, x):
        # x shape: [batch_size, 7, 20, 20]
        
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)  # [batch_size, 32, 10, 10]
        
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)  # [batch_size, 64, 5, 5]
        
        x = F.relu(self.conv3(x))  # [batch_size, 128, 5, 5]
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze()  # Return probabilities (0 to 1)


# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size = 8
    level = torch.randn(batch_size, 7, 20, 20)
    condition = torch.randint(0, 10, (batch_size,)).float()
    
    # Initialize models
    deep_q_model = DeepQCNN()
    classifier_model = MatrixClassifier()
    
    # Forward pass
    action = deep_q_model(level, condition)
    classification = classifier_model(level)
    
    print(f"Deep-Q Network Output Shape: {action.shape}")
    print(f"Classifier Output Shape: {classification.shape}")
    print(f"Classifier Output Sample: {classification}")