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
REPR_VERSION = 2


def find_bounding_box(matrix):
    """
    Find the bounding box of non-zero elements across all channels of an HWC matrix.
    
    Args:
        matrix: NumPy array with shape (height, width, channels)
        
    Returns:
        tuple: (top, left, bottom, right) coordinates of bounding box
    """
    # Check if matrix is in HWC format
    if len(matrix.shape) != 3:
        raise ValueError("Input must be a 3D array in HWC format")
    
    # Combine all channels using logical OR to find any non-zero pixels
    # This creates a 2D mask where a pixel is True if any channel has a non-zero value
    mask = np.any(matrix != 0, axis=2)
    
    # Find non-zero rows and columns
    non_zero_rows = np.where(np.any(mask, axis=1))[0]
    non_zero_cols = np.where(np.any(mask, axis=0))[0]
    
    # If no non-zero elements are found
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return (0, 0, 0, 0)
    
    # Get the bounding box coordinates
    top = non_zero_rows.min()
    bottom = non_zero_rows.max()
    left = non_zero_cols.min()
    right = non_zero_cols.max()
    
    return (top, left, bottom, right)


def get_entity_topleft(entity):
    # Uses Draftsman to get some data, returns topleft coords.
    from draftsman.data import entities as entity_data
    pos = entity['position']
    raw_entity = entity_data.raw.get(entity['name'])
    if raw_entity is None:
        return
    (l, t), (r, b) = raw_entity['selection_box']
    return (round(pos['x']+l),
            round(pos['y']+t))

def get_entity_bottomright(entity):
    # Uses Draftsman to get some data, returns topleft coords.
    from draftsman.data import entities as entity_data
    pos = entity['position']
    raw_entity = entity_data.raw.get(entity['name'])
    if raw_entity is None:
        return
    (l, t), (r, b) = raw_entity['selection_box']
    return (round(pos['x']+r),
            round(pos['y']+b))


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
    return blueprints

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

    t, l, b, r = find_bounding_box(matrix)
    if b - t > N:  # too tall!
        logging.error(f"Matrix is too tall, height of {h}, {b-t} but dims are {N}x{N}.")
        # return matrix
        return
    else:
        h = b - t
    if r - l > N:  # too wide!
        logging.error(f"Matrix is too wide, width of {w}, {r-l} but dims are {N}x{N}.")
        # return matrix
        return
    else:
        w = r - l
    
    # So, we need the top-left and bottom-right for the matrix.
    # 
    # Calculate start positions for centering
    start_h = (N - h) // 2
    start_w = (N - w) // 2

    matrix = matrix[t:b, l:r, :]
    
    # Place the original matrix in the center
    if c == 1:
        centered_matrix[start_h:start_h+h, start_w:start_w+w] = matrix.reshape(h, w)
    else:
        centered_matrix[start_h:start_h+h, start_w:start_w+w, :] = matrix
    
    return centered_matrix
    # return np.transpose(centered_matrix, (2, 0, 1))


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


@dataclass(frozen=False)
class Factory:
    json: dict
    _bp: Optional[Blueprint]=None
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
    
    # @lru_cache
    def get_matrix(self, dims: tuple[int, int],
                   repr_version: int=REPR_VERSION,
                   **kwargs) -> np.array:
        # Given a version, does some stuff. dims is w:h.
        if repr_version <= 1:
            # Basic 4 channel opacity.
            channels = ['assembler', 'inserter', 'belt', 'pole']
            mx = blueprint_to_opacity_matrices(self.blueprint, dims[0], dims[1], **kwargs)
        elif repr_version <= 2:
            # Opacity, power coverage, recipes, directionality.
            mx = json_to_6channel_matrix(self.json, dims[0], dims[1], **kwargs)
            # Need a recipe matrix too and a source/sink matrix.
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
    for m in jsf['blueprint']['entities']:
        a, b = get_entity_topleft(m)
        left = min(left, a)
        top = min(top, b)
    return left, top

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


# Define the index mapping.
from draftsman.data import entities

inserter_index = {k: ix+1 for ix, k in enumerate(entities.inserters)}
belt_index = {
    "transport-belt": 1,
    "fast-transport-belt": 2,
    "express-transport-belt": 3,
    "turbo-transport-belt": 4,
}
underground_index = {
    "underground-belt": 1,
    "fast-underground-belt": 2,
    "express-underground-belt": 3,
    "turbo-underground-belt": 4,
}
splitter_index = {
    "splitter": 1,
    "fast-splitter": 2,
    "express-splitter": 3,
    "turbo-splitter": 4,
}
belt_index.update(underground_index)
belt_index.update(splitter_index)
pole_index = {
    'small-electric-pole': 1,
    'medium-electric-pole': 2,
    'large-electric-pole': 3,
    'substation': 4
}
# Ignores the center point. So for each dimension, we can move that many squares away.
pole_radius = [2, 3, 1, 8]


def json_to_6channel_matrix(js: dict, w=None, h=None,
                                  trim_topleft: bool=True,
                                  center=False):
    # Creates opacity matrices AND directionality/electrical coverage matrix.
    entities = js['blueprint']['entities']
    channels = ['assembler', 'inserter', 'belt', 'pole', 'direction', 'power']
    if w is None or h is None:
        raise Exception("Must provide dimensions. (Sorry.)")
        pass  # break
        # w, h = bp.get_dimensions()

    if trim_topleft:
        left, top = bound_bp_by_json(js)
        for entity in entities:
            entity['position']['x'] -= left
            entity['position']['y'] -= top

    matrices = np.zeros((w+1, h+1, len(channels)), dtype=np.int8)

    # Setting dummy values.
    left, top = 100, 100
    right, bottom = -100, -100
    for entity in entities:
        width, height = 1, 1  # Default values.
        provides_power = False
        is_directional = False
        if entity['name'].startswith("assembling-machine"):
            # Assuming that assembling-machines end with their tier. Usually true.
            idx = int(entity['name'][-1])
            key = channels.index("assembler")
            width, height = 3, 3
        elif entity['name'] in inserter_index:
            is_directional = True
            idx = inserter_index[entity['name']]
            key = channels.index("inserter")
        elif entity['name'] in belt_index:
            is_directional = True
            idx = belt_index[entity['name']]
            key = channels.index("belt")
            if 'splitter' in entity['name']:
                if entity.get('direction', 0) in [0, 4]:
                    width = 2
                    height = 1
                else:
                    height = 2
                    width = 1
        elif entity['name'] in pole_index:
            provides_power = True
            if entity['name'] == 'substation':
                width, height = 2, 2
            idx = pole_index[entity['name']]
            key = channels.index("pole")
        else:
            logging.info(f"Skipping {entity['name']}.")
            entity = None
            continue

        logging.info(f"Placing {entity['name']}.")
        left, top = get_entity_topleft(entity)
        right = left + width
        bottom = top + height
        matrices[left:right, top:bottom, key] = idx
        logging.info(f"Taking up: {top}, {left} to {bottom}, {right}.")
        if is_directional:  # Do some stuff here.
            key = channels.index('direction')
            dir = entity.get('direction', 0)+1  # NOTE: I guess we have to offset here?
            matrices[left:right, top:bottom, key] = dir
        if provides_power:  # Do some more stuff here.
            key = channels.index('power')
            rad = pole_radius[idx-1]
            x0 = max(0, left-rad+1)
            x1 = min(w+1, right+rad-1)
            y0 = max(0, top-rad+1)
            y1 = min(h+1, bottom+rad-1) 
            logging.info(f"{entity['name']} with power {(x0, y0)}, {(x1, y1)}")

            matrices[x0:x1, y0:y1, key] = 1


    # NOTE: center_in_N doesn't seem to work.
    if center:
        N = max(w, h)
        matrices = center_in_N(matrix=matrices, N=N)
    return matrices
