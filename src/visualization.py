import matplotlib.pyplot as plt
import numpy as np
import torch

channel_list = ['assembler', 'inserter', 'belt', 'pole', 'direction', 'power']

def visualize_6D_matrix(matrix, hwc=None):
    """
    Visualize each channel of a multichannel 2D matrix in separate subplots.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        A matrix with shape (channels, height, width) or (height, width, channels)
    """
    # Determine the shape and reorder if necessary
    if matrix.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Check if channels are first or last dimension

    if hwc == False or matrix.shape[0] == 6:
        # (channels, height, width) format
        channels, height, width = matrix.shape
    elif hwc or matrix.shape[2] == 6:
        # (height, width, channels) format
        height, width, channels = matrix.shape
        # Transpose to (channels, height, width)
        matrix = np.transpose(matrix, (2, 0, 1))
    else:
        raise ValueError("Input must have exactly 6 channels")
    
    # Create figure and subplots with specific spacing
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), 
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
    axes = axes.flatten()
    
    # Plot each channel with exact pixel representation
    for i in range(channels):
        # Turn off interpolation for exact pixel representation
        im = axes[i].imshow(matrix[i], cmap='viridis', 
                           interpolation='none', aspect='equal')
        
        # Add precise grid lines at each pixel boundary
        height, width = matrix[i].shape
        
        # Draw vertical grid lines
        for x in range(width + 1):
            axes[i].axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
            
        # Draw horizontal grid lines
        for y in range(height + 1):
            axes[i].axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)
            
        # Set ticks at each integer position
        axes[i].set_xticks(np.arange(width))
        axes[i].set_yticks(np.arange(height))
        
        # Hide tick labels for cleaner appearance
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        
        # Make sure the limits are exactly at pixel boundaries
        axes[i].set_xlim(-0.5, width - 0.5)
        axes[i].set_ylim(height - 0.5, -0.5)  # Note: y-axis is inverted in imshow
        
        # Add border
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
            
        axes[i].set_title(f'{channel_list[i]}', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.suptitle('6-Channel Matrix Visualization', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    return fig


def visualize_7D_matrix(matrix, channel_list=None, hwc=None):
    """
    Visualize each channel of a multichannel 2D matrix in separate subplots.
   
    Parameters:
    -----------
    matrix : numpy.ndarray
        A matrix with shape (channels, height, width) or (height, width, channels)
    channel_list : list of str, optional
        Names for each channel to display as titles
    hwc : bool, optional
        If True, assumes input is (height, width, channels)
        If False, assumes input is (channels, height, width)
        If None, tries to determine automatically
    """
    # Default channel names if not provided
    if channel_list is None:
        channel_list = ["Assembler", "Inserter", "Belt", "Pole", 
                        "Direction", "Recipe", "Item"]
    
    # Determine the shape and reorder if necessary
    if matrix.ndim != 3:
        raise ValueError("Input must be a 3D array")
   
    # Check if channels are first or last dimension
    if hwc == False or matrix.shape[0] == 7:
        # (channels, height, width) format
        channels, height, width = matrix.shape
    elif hwc or matrix.shape[2] == 7:
        # (height, width, channels) format
        height, width, channels = matrix.shape
        # Transpose to (channels, height, width)
        matrix = np.transpose(matrix, (2, 0, 1))
    else:
        raise ValueError("Input must have exactly 7 channels")
   
    # Create figure and subplots with specific spacing
    fig, axes = plt.subplots(3, 3, figsize=(15, 15),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
    axes = axes.flatten()
   
    # Plot each channel with exact pixel representation
    for i in range(channels):
        # Turn off interpolation for exact pixel representation
        im = axes[i].imshow(matrix[i], cmap='viridis',
                           interpolation='none', aspect='equal')
       
        # Add colorbar for reference
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add precise grid lines at each pixel boundary
        height, width = matrix[i].shape
       
        # Draw vertical grid lines
        for x in range(width + 1):
            axes[i].axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
           
        # Draw horizontal grid lines
        for y in range(height + 1):
            axes[i].axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)
           
        # Set ticks at each integer position
        axes[i].set_xticks(np.arange(width))
        axes[i].set_yticks(np.arange(height))
       
        # Hide tick labels for cleaner appearance
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
       
        # Make sure the limits are exactly at pixel boundaries
        axes[i].set_xlim(-0.5, width - 0.5)
        axes[i].set_ylim(height - 0.5, -0.5)  # Note: y-axis is inverted in imshow
       
        # Add border
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
           
        axes[i].set_title(f'{channel_list[i]}', fontsize=12, pad=10)
   
    # Hide the unused subplot (9 plots for 7 channels)
    for i in range(channels, 9):
        axes[i].axis('off')
   
    plt.tight_layout()
    plt.suptitle('7-Channel Matrix Visualization', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Make room for the title
   
    return fig

def print_matrix_channels(matrix, channel_names=None):
    """
    Print each channel of a multi-channel matrix in a readable format.
    
    Args:
        matrix: Numpy array with shape (width, height, channels)
        channel_names: Optional list of names for each channel
    """
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(matrix.shape[2])]
    
    width, height, channels = matrix.shape
    
    for c in range(channels):
        print(f"\n=== {channel_names[c]} ===")
        
        # Print column headers
        print("    ", end="")
        for x in range(width):
            print(f"{x:2d} ", end="")
        print()
        
        # Print matrix with row headers
        for y in range(height):
            print(f"{y:2d}  ", end="")
            for x in range(width):
                val = matrix[x, y, c]
                if val == 0:
                    print(" . ", end="")
                else:
                    print(f"{val:2d} ", end="")
            print()



def visualize_channel_comparison(inputs, targets, outputs, num_samples=1, cmap='viridis'):
    """
    Visualize a comparison of 8-channel data for inputs, targets, and outputs.
    
    Args:
        inputs: Input tensor of shape [batch_size, 8, height, width]
        targets: Target tensor of shape [batch_size, 8, height, width]
        outputs: Output tensor of shape [batch_size, 8, height, width]
        num_samples: Number of samples to visualize
        cmap: Colormap to use for visualization
    """
    # Ensure all tensors are on CPU and converted to numpy
    if torch.is_tensor(inputs):
        inputs = inputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    
    # Column labels
    col_labels = ['Input', 'Ground Truth', 'Output']
    
    for sample_idx in range(min(num_samples, inputs.shape[0])):
        # Create an 8x3 grid plot with appropriate dimensions
        fig, axes = plt.subplots(8, 3, figsize=(12, 20))
        plt.subplots_adjust(wspace=0.05, hspace=0.3)
        
        # Set title for the figure
        fig.suptitle(f'Sample {sample_idx+1}: Channel Comparison', fontsize=16)
        
        # Add column labels at the top
        for col_idx, label in enumerate(col_labels):
            fig.text(0.2 + col_idx * 0.25, 0.96, label, ha='center', fontsize=14)
        
        # For each channel
        for channel in range(8):
            # Input channel
            im1 = axes[channel, 0].imshow(inputs[sample_idx, channel], cmap=cmap)
            axes[channel, 0].set_title(f'Channel {channel+1}', fontsize=10)
            axes[channel, 0].axis('off')
            
            # Ground truth channel
            im2 = axes[channel, 1].imshow(targets[sample_idx, channel], cmap=cmap)
            axes[channel, 1].axis('off')
            
            # Output channel
            im3 = axes[channel, 2].imshow(outputs[sample_idx, channel], cmap=cmap)
            axes[channel, 2].axis('off')
        
        # Add colorbar with better positioning
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # cbar = fig.colorbar(im1, cax=cbar_ax)
        # cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()