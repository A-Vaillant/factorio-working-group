import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BinaryMatrixTransformCNN(nn.Module):
    filename = 'binary_model'
    
    def __init__(self, matrix_size=20):
        super(BinaryMatrixTransformCNN, self).__init__()
        self.cnn = FactorioCellProcessorWithClamping(4, 4, 5, 3, 3, 2, 32)
        # Multiple conv layers with padding to maintain spatial dimensions
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Project back to original dimensions
        self.conv_out = nn.Conv2d(128, 21, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [batch, 21, 20, 20]
        x = self.cnn(x)
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
        x = self.cnn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        x = self.conv_out(x)
        return x

    def predict(self, X, threshold=0.5):
        """Make a prediction with thresholding"""
        self.eval()
        with torch.no_grad():
            # Get raw outputs
            raw_outputs = self(X)
            
            # Apply thresholding
            thresholded_outputs = (raw_outputs > threshold).float()
            
            return thresholded_outputs
        

# Full credit to Gemini Pro 2.5. My idea though.
# REPR_5: *[4, 4, 5, 3, 3, 2, 32]
class FactorioCellProcessorWithClamping(nn.Module):
    def __init__(self, C_opacity, C_dimensionality, C_recipe, C_item, C_kind, C_sourcesink,
                 projected_cell_features_dim):
        super().__init__()
        self.C_opacity = C_opacity
        self.C_dimensionality = C_dimensionality
        self.C_recipe = C_recipe
        self.C_item = C_item
        self.C_kind = C_kind
        self.C_sourcesink = C_sourcesink

        C_total_input = (C_opacity + C_dimensionality + C_recipe +
                         C_item + C_kind + C_sourcesink)

        self.cnn = nn.Sequential(
            nn.Conv2d(C_total_input, projected_cell_features_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, all_input_channels):
        opacity_ch = all_input_channels[:, :self.C_opacity, :, :]
        start_idx = self.C_opacity
        num_other_channels = (self.C_dimensionality + self.C_recipe + self.C_item +
                              self.C_kind + self.C_sourcesink)
        end_idx = start_idx + num_other_channels
        other_channels_original = all_input_channels[:, start_idx:end_idx, :, :]

        is_present_mask = (torch.sum(opacity_ch, dim=1, keepdim=True) > 0).float()
        other_channels_clamped = other_channels_original * is_present_mask
        processed_channels = torch.cat((opacity_ch, other_channels_clamped), dim=1)
        cell_features = self.cnn(processed_channels)
        return cell_features

class UNetConvBlock(nn.Module): # Helper for U-Net like structure
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        # If using batchnorm, bias in Conv2d is redundant
        conv_bias = not use_batchnorm

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class FactorioCNN_PixelOutput(nn.Module):
    filename = 'factorio_unet'

    def __init__(self, C_opacity, C_dimensionality, C_recipe, C_item, C_kind, C_sourcesink,
                 projected_cell_features_dim, num_output_channels, use_batchnorm=True,
                 presence_gate_strength=1.0, # For tuning the presence gate
                 presence_gate_bias=0.0):      # For tuning the presence gate
        super().__init__()
        self.C_opacity = C_opacity
        self.dependent_dims =  [C_dimensionality, C_recipe, C_item, C_kind, C_sourcesink]
        self.presence_gate_strength = presence_gate_strength
        self.presence_gate_bias = presence_gate_bias
        self.cnn = FactorioCellProcessorWithClamping(
            C_opacity, C_dimensionality, C_recipe, C_item, C_kind, C_sourcesink,
            projected_cell_features_dim
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = UNetConvBlock(projected_cell_features_dim, 64, use_batchnorm) # 20x20 -> 20x20
        self.enc2 = UNetConvBlock(64, 128, use_batchnorm)                           # 10x10 -> 10x10
        self.enc3 = UNetConvBlock(128, 256, use_batchnorm)                          # 5x5   -> 5x5
        
        # Bottleneck (or deepest part of encoder)
        # For 20x20, going deeper might make spatial dim too small (e.g. 2x2)
        # Let's make enc3 the bottleneck for simplicity here given 20x20.
        # If a deeper bottleneck is desired, ensure pooling/convs handle small dimensions.

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 5x5 -> 10x10
        self.dec2 = UNetConvBlock(256, 128, use_batchnorm) # 128 (up) + 128 (skip) = 256

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 10x10 -> 20x20
        self.dec1 = UNetConvBlock(128, 64, use_batchnorm)   # 64 (up) + 64 (skip) = 128

        # Final output convolution
        self.final_conv = nn.Conv2d(64, num_output_channels, kernel_size=1)

    def forward(self, concatenated_input_channels):
        # Initial per-cell processing
        x0 = self.cnn(concatenated_input_channels) # (B, projected_dim, 20, 20)

        # Encoder
        e1_conv = self.enc1(x0)      # (B, 64, 20, 20)
        e1_pool = self.pool(e1_conv) # (B, 64, 10, 10)

        e2_conv = self.enc2(e1_pool) # (B, 128, 10, 10)
        e2_pool = self.pool(e2_conv) # (B, 128, 5, 5)
        
        bottleneck = self.enc3(e2_pool) # (B, 256, 5, 5)

        # Decoder
        d2_up = self.upconv2(bottleneck)                       # (B, 128, 10, 10)
        d2_cat = torch.cat([d2_up, e2_conv], dim=1)            # (B, 128+128, 10, 10)
        d2_conv = self.dec2(d2_cat)                            # (B, 128, 10, 10)

        d1_up = self.upconv1(d2_conv)                          # (B, 64, 20, 20)
        d1_cat = torch.cat([d1_up, e1_conv], dim=1)            # (B, 64+64, 20, 20)
        d1_conv = self.dec1(d1_cat)                            # (B, 64, 20, 20)

        raw_output_logits = self.final_conv(d1_conv)                      # (B, num_output_channels, 20, 20)
        
        # --- Apply architectural soft gating for Rule 1 (Opacity) ---
        current_idx = 0
        opacity_logits = raw_output_logits[:, current_idx : current_idx + self.C_opacity, :, :]
        current_idx += self.C_opacity
        
        dependent_logits = raw_output_logits[:, current_idx:, :, :]
        # Generate a presence gate from opacity logits
        # Assuming "Interpretation A": if any of the 4 opacity types has a high logit, entity is present.
        # We take the max logit across the specific entity types as an indicator of presence.
        # If your opacity_logits include a "no entity" channel, adjust this logic.
        presence_score = torch.max(opacity_logits, dim=1, keepdim=True)[0] # Max logit for any entity type
        # Scale and shift before sigmoid to control steepness/threshold of the gate
        presence_gate = torch.sigmoid(presence_score * self.presence_gate_strength - self.presence_gate_bias) 

        # Apply soft gate: if presence_gate is near 0, dependent logits are suppressed.
        gated_dependent_logits = dependent_logits * presence_gate
        
        final_output_logits = torch.cat((opacity_logits, gated_dependent_logits), dim=1)
        
        return final_output_logits
    
    def predict(self, X):
        """Make a prediction and postprocess it."""
        self.eval()
        with torch.no_grad():
            # Forward pass with constraints
            outputs = self(X)
            
            processed_outputs = self._post_process(output_logits=outputs, logit_presence_threshold=0.0)
            
            return processed_outputs
        
    def _post_process(self, output_logits, logit_presence_threshold=0.0,
                             # Threshold for deciding presence if using individual sigmoid activations
                             activation_presence_threshold=0.5,
                             opacity_threshold=0.5): # Threshold to decide presence from opacity probabilities
        final_consistent_output = torch.zeros_like(output_logits)
        current_idx = 0
        
        C_opacity_out = self.C_opacity
        C_dim_out, C_recipe_out, C_item_out, C_kind_out, C_ss_out = self.dependent_dims
        final_predictions = output_logits.clone() # Start with raw logits or probabilities

        # --- Interpret Opacity and derive a HARD presence mask ---
        opacity_logits = final_predictions[:, :C_opacity_out, :, :]       
        max_logit_val, _ = torch.max(opacity_logits, dim=1, keepdim=True)
        is_present_hard_mask = (max_logit_val > logit_presence_threshold).float()

        # --- Rule 1: Opacity Clamping (Hard) ---
        # If not present, zero out dependent channels
        opacity_probs_for_type_selection = torch.softmax(opacity_logits, dim=1)
        max_opacity_prob = torch.argmax(opacity_probs_for_type_selection, dim=1, keepdim=True)
        opacity_one_hot = torch.zeros_like(opacity_probs_for_type_selection).scatter_(1, max_opacity_prob, 1.0)
        # Store this one-hot opacity, but only where an entity is present
        final_consistent_output[:, current_idx : current_idx + C_opacity_out, :, :] = opacity_one_hot * is_present_hard_mask
        current_idx = C_opacity_out
        
        # One-hot for the rest of the variables too.
        for dim in self.dependent_dims:
            logits = final_predictions[:, current_idx:current_idx+dim, :, :]
            probs = torch.softmax(logits, dim=1)
            max_prob = torch.argmax(probs, dim=1, keepdim=True)
            one_hot = torch.zeros_like(probs).scatter_(1, max_prob, 1.0)
            final_consistent_output[:, current_idx:current_idx+dim, :, :] = one_hot * is_present_hard_mask
            current_idx += dim
        
        # --- Rule 2: Object <-> Item ID (Conceptual) ---
        # This requires deciding what "NEEDS an item_id" means for your output (e.g., a specific item_id channel must be active).
        # Example: If is_present_hard_mask is 1, but item_preds for that cell are all "off" (e.g., after softmax, "no_item_class" is max),
        # you might force a default item or flag inconsistency.
        # And vice-versa: if item_preds are "on" but is_present_hard_mask became 0, Rule 1 already zeroed item_preds.

        # --- Rule 3: Recipe -> Assembler (Conceptual) ---
        # Example: Determine if a recipe is active (e.g., max recipe_prob > threshold_recipe_active)
        # recipe_probs = torch.softmax(recipe_preds, dim=1) # Assuming recipe_preds are logits
        # is_recipe_active_mask = (torch.max(recipe_probs, dim=1, keepdim=True)[0] > 0.5).float() # Example
        #
        # is_assembler_present = (predicted_entity_type_idx == assembler_type_idx).float() * is_present_hard_mask
        #
        # # If recipe active BUT assembler not present:
        # violation_mask_rule3 = is_recipe_active_mask * (1 - is_assembler_present)
        # if torch.any(violation_mask_rule3 > 0):
        #     # Option 1: Force assembler type (might conflict with other types)
        #     # opacity_logits_updated = ... (ensure assembler_type_idx logit is high, others low)
        #     # Option 2: Nullify recipe if assembler cannot be forced
        #     # recipe_preds_updated = recipe_preds * (1 - violation_mask_rule3) # Zero out recipe where violated
        #     pass # Implement chosen conflict resolution

        return final_consistent_output


class FactorioCNN_Repr5(FactorioCNN_PixelOutput):
    def __init__(self, *args, **kwargs):
        super().__init__(
            4, 4, 5, 3, 3, 2, 32, 21, *args, **kwargs)
