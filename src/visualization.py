import numpy as np
import matplotlib.pyplot as plt


channel_list = ['assembler', 'inserter', 'belt', 'pole', 'direction', 'power']

def visualize_multichannel_matrix(matrix, hwc=None):
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