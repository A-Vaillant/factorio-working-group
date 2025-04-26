""" representation.py

Takes a Blueprint object and returns something you can do some good old fashioned machine learning on."""

import numpy as np
import math
from draftsman.blueprintable import Blueprint, Blueprintable, get_blueprintable_from_JSON, get_blueprintable_from_string
from draftsman.utils import string_to_JSON
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import logging

# You surely will not regret putting a global variable in your representation module.
REPR_VERSION = 1


def recursive_json_parse(json_bp: dict) -> list[dict]:
    if 'blueprint' in json_bp:
        # it's a blueprint!
        return [json_bp]
    # it's not.
    elif not 'blueprint_book' in json_bp:
        # It's something ELSE. SKIP IT.
        return []
    blueprints = []
    for bp in json_bp['blueprint_book']['blueprints']:
        blueprints += recursive_json_parse(bp)
    return [blueprints]

def recursive_blueprint_book_parse(bp_book: Blueprintable) -> list[Blueprint]:
    # Reached a leaf.
    if isinstance(bp_book, Blueprint):
        return [bp_book]

    blueprints = []
    for bp_node in bp_book.blueprints:
        blueprints += recursive_blueprint_book_parse(bp_node)
    return blueprints

def center_in_N(matrix, N: int = 15) -> np.ndarray[np.ndarray]:
    """
    Center a matrix of any size within an NxN matrix (numpy array) and converts
    it to (C, H, W).
    (Should be two separate functions.)
    
    Args:
        matrix: Input numpy array of shape (H, W) or (H, W, C)
    
    Returns:
        centered_matrix: NxN matrix with the input matrix centered, of shape (C, H, W).
    """
    # Get input matrix dimensions
    if matrix.ndim == 2:
        h, w = matrix.shape
        c = 1
        matrix = matrix.reshape(h, w, 1)
    else:
        h, w, c = matrix.shape
    
    # Create empty NxN matrix
    if c == 1:
        centered_matrix = np.zeros((N, N))
    else:
        centered_matrix = np.zeros((N, N, c))
    
    # Calculate start positions for centering
    start_h = (N - h) // 2
    start_w = (N - w) // 2
    
    # Place the original matrix in the center
    if c == 1:
        centered_matrix[start_h:start_h+h, start_w:start_w+w] = matrix.reshape(h, w)
    else:
        centered_matrix[start_h:start_h+h, start_w:start_w+w, :] = matrix
    
    # 
    return np.transpose(centered_matrix, (2, 0, 1))

def map_entity_to_key(entity) -> str:
    key = None
    # Determine which matrix to use based on entity type (using string comparison)
    if entity.type == "assembling-machine":
        key = "assembler"
    elif entity.type == "inserter":
        key = "inserter"
    elif entity.type in ["transport-belt", "splitter", "underground-belt"]:
        key = "belt"
    elif entity.type in ["electric-pole"]:
        key = "pole"
    else:
        pass
    return key

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
    
    # note: you can just do list[cls] in Python 3.10+. a thing to consider in setting dependencies.
    @classmethod
    def from_blueprintable(cls, input: str) -> list["Factory"]:
        # j = string_to_JSON(input)
        j = get_blueprintable_from_string(input)  # Blueprintable
        j_s = recursive_blueprint_book_parse(j)   # list[Blueprint]
        return [cls.from_blueprint(j_) for j_ in j_s]  # Creates a list of Factories.

    @property
    def blueprint(self) -> Blueprint:
        if self._bp is None:
            self._bp = get_blueprintable_from_JSON(self.json)
        return self._bp
    
    @lru_cache
    def get_matrix(self, dims: tuple[int, int],
                   repr_version: int=REPR_VERSION,
                   **kwargs) -> np.array:
        # Given a version, does some stuff. dims is w:h.
        if repr_version >= 1:
            # Basic 4 channel opacity.
            channels = ['assembler', 'inserter', 'belt', 'pole']
            mx = blueprint_to_opacity_matrices(self.blueprint, dims[0], dims[1], **kwargs)
        elif repr_version >= 2:
            # Opacity, power coverage, recipes, directionality.
            mx = None
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

def bound_bp_by_json(jsf: dict):
    """Returns the top-left most point that defines the minimal
    rectangle. In a compacted matrix, this returns (0,0).

    Args:
        jsf (dict): The JSON representation of a Factory.
    """
    left = 100
    top = 100
    raise NotImplementedError("Have not implemented this function.")
    pass

def blueprint_to_opacity_matrices(bp: Blueprint, w=None, h=None,
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