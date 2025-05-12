import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BinaryMatrixTransformCNN(nn.Module):
    filename = 'binary_model'
    
    def __init__(self, channels=21, matrix_size=20):
        super(BinaryMatrixTransformCNN, self).__init__()
        
        # Multiple conv layers with padding to maintain spatial dimensions
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
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
        

class ConstrainedBinaryMatrixTransformCNN(nn.Module):
    filename = 'constrained_binary_model'

    def __init__(self, channels=21, matrix_size=20): # matrix_size is not used in this specific PyTorch module code
        super().__init__()

        # Define channel structure for constraints
        self.channel_slices = {
            'assembler': slice(0, 1),
            'belt': slice(1, 2),
            'inserter': slice(2, 3),
            'pole': slice(3, 4),
            'direction': slice(4, 8),    # 4 channels
            'recipe': slice(8, 13),     # 5 channels
            'item': slice(13, 16),      # 3 channels
            'kind': slice(16, 19),      # 3 channels
            'sourcesink': slice(19, 21), # 2 channels
        }

        # Required fields for each entity type
        # Meaning: If 'assembler' is active, 'item' must also be meaningfully active.
        # This will modulate the entity's own activation: P(entity_final) = P(entity_current) * P(all_requirements_present)
        self.required_fields = {
            'assembler': ['item'],
            'belt': ['item', 'direction', 'kind'],
            'inserter': ['item', 'direction'],
            'pole': ['item'],
        }

        # Fields that can only be active if specific parent entities are active
        # Meaning: P(dependent_field_final) = P(dependent_field_current) * P(any_parent_active)
        self.dependent_fields = {
            'recipe': ['assembler'],
            'direction': ['belt', 'inserter'],
            'kind': ['belt'],
            # 'sourcesink' might also depend on 'belt', but not listed.
            # 'item' is not listed as dependent on any primary entity, assuming it can be predicted independently
            # and then used by 'required_fields' logic.
        }

        # Categorical fields that should be one-hot encoded (via softmax)
        # Other fields (not listed here) will be treated as independent binary channels (via sigmoid)
        self.categorical_fields = ['direction', 'recipe', 'item', 'kind', 'sourcesink']

        # Base CNN architecture
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Project back to original dimensions (number of channels)
        self.conv_out = nn.Conv2d(128, channels, kernel_size=1)

    def _extract_features(self, x):
        """Standard feature extraction part of the CNN."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def _apply_constraints(self, raw_output_logits):
        # This dictionary will store P(field_is_on_at_all_at_this_pixel)
        field_on_activations = {}
        # This tensor will store the "final" channel-wise activations.
        # For categorical, it's P(category_i | field_is_on) * P(field_is_on)
        # For binary, it's just P(field_is_on)
        final_channel_activations = torch.zeros_like(raw_output_logits)

        # --- Step 1: Initial Field "On-ness" and Category Probabilities ---
        # For categorical fields, P(category_i | field_is_on_hypothetically).
        # For binary fields, this is not needed as P_on is the activation.
        initial_category_probs = {} 

        all_field_names = list(self.channel_slices.keys())
        for field_name in all_field_names:
            field_slice = self.channel_slices[field_name]
            if field_name in self.categorical_fields:
                probs = F.softmax(raw_output_logits[:, field_slice, :, :], dim=1)
                initial_category_probs[field_name] = probs
                # P_on for categorical fields is initialized to 1.0;
                # it will be scaled down if its parents (from dependent_fields) are off.
                # If it has no parents, it remains 1.0 (unless it's a primary entity itself, which isn't the case for categoricals here).
                field_on_activations[field_name] = torch.ones_like(raw_output_logits[:, 0:1, :, :])
            else: # Binary fields (primary entity opacities)
                p_on = torch.sigmoid(raw_output_logits[:, field_slice, :, :])
                field_on_activations[field_name] = p_on
                # For binary fields, initial_category_probs is not applicable in the same way.

        # --- Step 2: Update Field "On-ness" based on Dependencies ---
        # This pass calculates the P_on for each field after considering its parents.
        # It's important to compute all S2 P_on values based on S1 P_on values before moving to S3.
        field_on_activations_s2 = field_on_activations.copy() # Use S1 values as base for S2 calculations

        for field_name in all_field_names: # Iterate to ensure all fields get an S2 P_on
            if field_name in self.dependent_fields:
                parent_entity_names = self.dependent_fields[field_name]
                
                # P_on for this field was initialized (e.g., to 1.0 if categorical, or sigmoid if binary parent)
                # This initial P_on is what gets modulated.
                initial_p_on_this_field_s1 = field_on_activations[field_name] 

                combined_parent_p_on_s1 = torch.zeros_like(raw_output_logits[:, 0:1, :, :])
                for parent_name in parent_entity_names:
                    # Parents' P_on values are from the initial field_on_activations (S1)
                    parent_p_on = field_on_activations[parent_name] 
                    combined_parent_p_on_s1 = torch.max(combined_parent_p_on_s1, parent_p_on)
                
                # Update P_on for field_name for S2
                field_on_activations_s2[field_name] = initial_p_on_this_field_s1 * combined_parent_p_on_s1
            # Else: if field_name is not in dependent_fields, its P_on from S1 carries over to S2.
            # This is already handled by field_on_activations_s2 = field_on_activations.copy()

        # --- Step 3: Update Field "On-ness" based on Requirements ---
        # This pass further refines P_on for entities based on their required children's S2 P_on.
        field_on_activations_s3 = field_on_activations_s2.copy()

        for entity_name in all_field_names: # Iterate to ensure all fields get an S3 P_on
            if entity_name in self.required_fields:
                req_field_names_list = self.required_fields[entity_name]
                
                p_on_entity_s2 = field_on_activations_s2[entity_name] # Entity's current P_on
                
                product_of_req_p_on_s2 = torch.ones_like(p_on_entity_s2)
                for req_field_name in req_field_names_list:
                    # P_on of the required child field (after its own dependencies were processed in S2)
                    p_on_req_field_s2 = field_on_activations_s2[req_field_name] 
                    product_of_req_p_on_s2 = product_of_req_p_on_s2 * p_on_req_field_s2
                
                # Update P_on for entity_name for S3
                field_on_activations_s3[entity_name] = p_on_entity_s2 * product_of_req_p_on_s2
            # Else: P_on from S2 carries over to S3.

        # --- Step 4: Construct Final Channel Activations ---
        # Final P(channel_i) = P(category_i | field_on) * P_on_s3(field) for categoricals
        # Final P(channel_i) = P_on_s3(field) for binary fields
        for field_name in all_field_names:
            field_slice = self.channel_slices[field_name]
            p_on_final = field_on_activations_s3[field_name]
            if field_name in self.categorical_fields:
                # initial_category_probs[field_name] is P(category_i | field_on_hypothetically)
                final_channel_activations[:, field_slice, :, :] = \
                    initial_category_probs[field_name] * p_on_final.expand_as(initial_category_probs[field_name])
            else: # Binary fields
                final_channel_activations[:, field_slice, :, :] = p_on_final
                
        return final_channel_activations

    def forward(self, x):
        features = self._extract_features(x)
        raw_output_logits = self.conv_out(features)
        constrained_output = self._apply_constraints(raw_output_logits)
        return constrained_output
    
    def predict(self, X):
        """Make a prediction with thresholding"""
        self.eval()
        with torch.no_grad():
            # Forward pass with constraints
            outputs = self(X)
            
            # Apply thresholding
            thresholded_outputs = (outputs > 0.5).float()
            
            return thresholded_outputs


class ConstrainedAttentiveBinaryMatrixTransformCNN(ConstrainedBinaryMatrixTransformCNN):
    filename = 'constrained_attention_model'

    def __init__(self, channels=21, matrix_size=20):
        super().__init__(channels, matrix_size)
        
        self.attention1 = SelfAttention(64)
        self.attention2 = SelfAttention(128)
        self.attention3 = SelfAttention(128)
    
    def _extract_features(self, x):
        """Override feature extraction to include attention"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        return x