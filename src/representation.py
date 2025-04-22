""" representation.py

Takes a Blueprint object and returns something you can do some good old fashioned machine learning on."""

import numpy as np
import math
from draftsman.blueprintable import Blueprint, Blueprintable, get_blueprintable_from_JSON
from draftsman.utils import string_to_JSON
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import logging


# You surely will not regret putting a global variable in your representation module.
REPR_VERSION = 1


class RepresentationError(ValueError):
    """Raised when there's an issue with creating or maintaining a Factory representation."""
    pass


class Factory:
    _bp: Optional[Blueprint]=None
    json: dict
    # matrix: Optional[np.array]=None

    @classmethod
    def from_blueprint(cls, input: Blueprint):
        j = input.to_dict()
        if 'blueprint' not in j:
            raise RepresentationError("Passed in something that wasn't a blueprint.")
        return Factory(_bp=input, json=j)

    @classmethod
    def from_str(cls, input: str):
        j = string_to_JSON(input)
        if 'blueprint' not in j:  # not a proper blueprint
            if 'blueprint-book' in j:  # this is a BOOK
                raise RepresentationError("Attempted to represent a blueprint book as a Factory.")
            else:
                logging.debug(j)
                raise RepresentationError("Failed for an unknown issue.")
        return Factory(json=j)

    @property
    def blueprint(self) -> Blueprint:
        if self._bp is None:
            bp = get_blueprintable_from_JSON(self.json)
        return self._bp
    
    @lru_cache
    def get_matrix(self, dims: tuple[int, int],
                   repr_version: int=REPR_VERSION,
                   **kwargs) -> np.array:
        # Given a version, does some stuff. dims is w:h.
        if repr_version > 0:
            # Basic 4 channel opacity.
            channels = ['assembler', 'inserter', 'belt', 'pole']
            mx = blueprint_to_matrices(self.blueprint, dims[0], dims[1], **kwargs)
        return mx




def trim_zero_edges(matrix,
                    channel_selector=lambda channels: True):
    """
    Trims edge rows/columns of a matrix that are all zeros across channels that satisfy
    the channel_selector function.
    
    Args:
        matrix: numpy array of shape (H, W, C) or (H, W)
        channel_selector: function that takes a channel index and returns True or False
                         to indicate whether that channel should be considered when
                         checking for zero values
    
    Returns:
        Trimmed matrix with zero edges removed
    """
    # Handle case of 2D matrix (single channel)
    if matrix.ndim == 2:
        matrix = matrix[..., np.newaxis]
    
    # Get dimensions
    height, width, channels = matrix.shape
    
    # Create masks for the channels we care about
    channel_mask = np.array([channel_selector(c) for c in range(channels)], dtype=bool)
    
    # If no channels selected, return original matrix
    if not np.any(channel_mask):
        return matrix
    
    # Find non-zero rows (any non-zero value in selected channels)
    row_has_data = np.any(matrix[:, :, channel_mask] != 0, axis=(1, 2))
    col_has_data = np.any(matrix[:, :, channel_mask] != 0, axis=(0, 2))
    
    # Find first and last non-zero row/column
    first_row = np.argmax(row_has_data)
    last_row = height - np.argmax(row_has_data[::-1]) if np.any(row_has_data) else 0
    
    first_col = np.argmax(col_has_data)
    last_col = width - np.argmax(col_has_data[::-1]) if np.any(col_has_data) else 0
    
    # Return trimmed matrix (handle empty case)
    if first_row >= last_row or first_col >= last_col:
        # All zeros in selected channels, return minimal slice
        return matrix[0:1, 0:1]
    
    result = matrix[first_row:last_row, first_col:last_col]
    
    # Return 2D array if input was 2D
    if matrix.shape[2] == 1 and matrix.ndim == 3:
        return result.squeeze(axis=2)
    
    return result

def bound_bp_by_entities(bbook: Blueprint):
    left = 100
    top = 100
    for m in bbook.entities:
        left = min(left, m.tile_position._data[0])
        top = min(top, m.tile_position._data[1])
    return left, top



def blueprint_to_matrices(bp, w=None, h=None,
                          trim_topleft: bool=True):
    """
    Convert a Factorio blueprint to multiple binary matrices representing entity types.
    
    Args:
        bp: A factorio_draftsman Blueprint object
        
    Returns:
        Dictionary of numpy arrays, one per entity type
    """
    # Find blueprint bounding box to center it
    if w is None or h is None:
        w, h = bp.get_dimensions()

    if trim_topleft:
        left, top = bound_bp_by_entities(bp)
        bp.translate(-left, -top)

    channels = ['assembler', 'inserter', 'belt', 'pole']
    matrices = np.zeros((w+1, h+1, len(channels)), dtype=np.int8)

    # Place entities in matrices
    for entity in bp.entities:
        key = None
        width, height = None, None
        # Determine which matrix to use based on entity type (using string comparison)
        if entity.type == "assembling-machine":
            key = channels.index("assembler")
            width, height = 3, 3
        elif entity.type == "inserter":
            key = channels.index("inserter")
        elif entity.type in ["transport-belt", "splitter", "underground-belt"]:
            key = channels.index("belt")
            if entity.type == 'splitter':
                if entity.direction in [0, 4]:
                    width = 2
                    height = 1
                else:
                    height = 2
                    width = 1
        elif entity.type in ["electric-pole"]:
            key = channels.index("pole")
        else:
            continue
        
        width = width or 1
        height = height or 1

        left, top = entity.tile_position._data
        right = left + width
        bottom = top + height
        matrices[left:right, top:bottom, key] = 1
    
    return matrices