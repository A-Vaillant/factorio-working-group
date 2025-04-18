""" representation.py

Takes a Blueprint object and returns something you can do some good old fashioned machine learning on."""

import numpy as np
import math
from draftsman.blueprintable import Blueprint

def blueprint_to_matrices(bp, grid_size=25):
    """
    Convert a Factorio blueprint to multiple binary matrices representing entity types.
    
    Args:
        bp: A factorio_draftsman Blueprint object
        grid_size: Size of the grid (default: 25x25)
        
    Returns:
        Dictionary of numpy arrays, one per entity type
    """
    # Find blueprint bounding box to center it
    w, h = bp.get_dimensions()
    if w > grid_size or h > grid_size:
        raise Exception("Blueprint too big!")
    aabb = bp.get_world_bounding_box()
    aa = aabb.top_left
    print(f"{aa} is aa")
    # Returns central point as integers.
    blueprint_center = (int(aa[0] + w/2), int(aa[1] + h/2))
    print(blueprint_center)
    blueprint_center = (0, 0)  # NO
    # Moves the blueprint so that it's aligned with the grid.
    grid_translate = (grid_size // 2, grid_size // 2)
    total_translate = (grid_translate[0] - blueprint_center[0],
                       grid_translate[1] - blueprint_center[1])
    bp.translate(*total_translate)

    # Initialize matrices for different entity types
    matrices = dict()
    for k in ['assemblers', 'inserters', 'belts', 'electric_poles']:
        matrices[k] = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Place entities in matrices
    for entity in bp.entities:
        print(f"placing {entity.type}")
        width = entity.tile_width
        height = entity.tile_height
        aa, bb = entity.tile_position, entity.tile_position + (entity.tile_width, entity.tile_height)
        print(f"top left is {aa}, bot right is {bb}. width is {width}, height is {height}")
        key = None
        # Determine which matrix to use based on entity type (using string comparison)
        if entity.type == "assembling-machine":
            key = "assemblers"
        elif entity.type == "inserter":
            key = "inserters"
        elif entity.type in ["transport-belt", "splitter", "underground-belt"]:
            key = "belts"
        elif entity.type in ["electric-pole"]:
            key = "electric_poles"
        else:
            continue
        
        left, top = aa
        right, bottom = bb
        # Set the corresponding cells to 1
        matrices[key][top:bottom, left:right] = 1
        # for i in range(left, right + 1):
        #     for j in range(top, bottom + 1):
        #         if 0 <= i < grid_size and 0 <= j < grid_size:
        #             matrices[key][j, i] = 1
        #         else:
        #             raise Exception("Out of bounds.")
    
    return matrices


# def blueprint_to_tensor(bp, grid_size=25):
#     """
#     Convert blueprint to a single tensor with multiple channels.
    
#     Args:
#         bp: A factorio_draftsman Blueprint object
#         grid_size: Size of the grid (default: 25x25)
        
#     Returns:
#         Numpy array of shape (5, grid_size, grid_size)
#     """
#     matrices = blueprint_to_matrices(bp, grid_size)
    
#     # Stack matrices into a single tensor
#     tensor = np.stack([
#         matrices["assemblers"],
#         matrices["inserters"],
#         matrices["belts"],
#         matrices["electric_poles"],
#         matrices["other"]
#     ], axis=0)
    
#     return tensor