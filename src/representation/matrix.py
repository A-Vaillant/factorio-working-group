""" matrix.py
Contains the representation code for numpy matrices.
"""
import numpy as np
import logging
from draftsman.data import entities
from draftsman.blueprintable import Blueprint

from .utils import (bound_bp_by_entities,
                    bound_bp_by_json,
                    get_entity_topleft,
                    get_item_id_from_name,
                    RepresentationError)


# You surely will not regret putting a global variable in your representation module.
global REPR_VERSION
REPR_VERSION = 5

# AV: This is incomplete, to reduce the complexity involved.
# inserter_index = {k: ix+1 for ix, k in enumerate(entities.inserters)}
inserter_index = {
    "inserter": 1,
    "fast-inserter": 2,
    "long-handed-inserter": 3
}

belt_index = {
    "transport-belt": 1,
    "fast-transport-belt": 2,
    "express-transport-belt": 3,
    # "turbo-transport-belt": 4,
}
underground_index = {
    "underground-belt": 1,
    "fast-underground-belt": 2,
    "express-underground-belt": 3,
    "ee-infinity-loader": 1,
    # "turbo-underground-belt": 4,
}
splitter_index = {
    "splitter": 1,
    "fast-splitter": 2,
    "express-splitter": 3,
    # "turbo-splitter": 4,
}
belt_index.update(underground_index)
belt_index.update(splitter_index)
pole_index = {
    'small-electric-pole': 1,
    'medium-electric-pole': 2,
    'large-electric-pole': 3,
    # 'substation': 4
}
# Ignores the center point. So for each dimension, we can move that many squares away.
pole_radius = [2, 3, 1, 8]

# ----------- Utilities. ------------------
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
        matrix = matrix.reshape(h, w, c)
    else:
        h, w, c = matrix.shape
    
    # Create empty NxN matrix
    centered_matrix = np.zeros((N, N, c), dtype=int)

    t, l, b, r = find_bounding_box(matrix)
    # logging.info(f"Bounding box: top={t}, left={l}, bottom={b}, right={r}")
    # Calculate content dimensions
    content_h = b - t + 1
    content_w = r - l + 1
    
    # Validate size
    if content_h > N or content_w > N:
        logging.error(f"Content size {content_h}x{content_w} exceeds target size {N}x{N}")
        return None
    
    # So, we need the top-left and bottom-right for the matrix.
    # 
    # Calculate start positions for centering
    start_h = (N - content_h) // 2
    start_w = (N - content_w) // 2

    content = matrix[t:b+1, l:r+1, :]
    # logging.info(f"Extracted content shape: {content.shape}")
    centered_matrix[start_h:start_h+content_h, start_w:start_w+content_w, :] = content

    # Place the original matrix in the center
    if c == 1:
        centered_matrix = centered_matrix[:, :, 1]  # Return a 2x2 matrix for funsies
        # centered_matrix[start_h:start_h+content_h, start_w:start_w+content_w] = centered_matrix.reshape(h, w)
    return centered_matrix


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

# ----------- Representations. ------------
# REPR_VERSION = 1
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

# REPR_VERSION = 2
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

# REPR_VERSION = 3
def json_to_7channel_matrix(js: dict, w, h,
                            trim_topleft: bool=True,
                            center=False):
    # Creates opacity matrices, directionality, recipe and item ID channels.
    entities = js['blueprint']['entities']
    channels = ['assembler', 'inserter', 'belt', 'pole', 'direction', 'recipe', 'item']

    if trim_topleft:
        left, top = bound_bp_by_json(js)
        for entity in entities:
            entity['position']['x'] -= left
            entity['position']['y'] -= top

    mats = np.zeros((w + 1, h + 1, len(channels)), dtype=np.int16)

    # build the recipe‑id map once
    recipe_id = _make_recipe_index(entities)
    
    for entity in entities:
        name, width, height = entity['name'], 1, 1  # Default values.
        is_directional = False
        idx = get_item_id_from_name(name)
        if name.startswith("assembling-machine"):
            key = channels.index("assembler")
            width, height = 3, 3
        elif name in inserter_index:
            key = channels.index("inserter")
            is_directional = True
        elif name in belt_index:
            is_directional = True
            key = channels.index("belt")
            if 'splitter' in name:
                if entity.get('direction', 0) in [0, 4]:
                    width, height = 2, 1
                else:
                    width, height = 1, 2
        elif name in pole_index:
            if name == 'substation':
                width, height = 2, 2
            key = channels.index("pole")
        else:
            logging.info(f"Skipping {entity['name']}.")
            entity = None
            continue

        # Placement logic
        logging.info(f"Placing {entity['name']}.")
        # Determining the item's bounds.
        lx, ty = get_entity_topleft(entity)
        rx, by = lx + width, ty + height
        # Placing the building on the matrix.
        mats[lx:rx, ty:by, key] = 1
        mats[lx:rx, ty:by, channels.index('item')] = idx
        
        logging.info(f"Taking up: {ty}, {lx} to {by}, {rx}.")
        if is_directional:
            mats[lx:rx, ty:by, channels.index("direction")] = entity.get('direction', 0) + 1

        # recipe
        if key == 0 and "recipe" in entity:
            mats[lx:rx, ty:by, channels.index("recipe")] = recipe_id[entity["recipe"]]

    # NOTE: center_in_N doesn't seem to work.
    if center:
        N = max(w, h)
        mats = center_in_N(matrix=mats, N=N)
    return mats

# REPR_VERSION = 4
def json_to_8channel_matrix(js: dict, w, h,
                            trim_topleft: bool=True,
                            center=False):
    # Creates opacity matrices, directionality, recipe, item ID and source/sink channels.
    entities = js['blueprint']['entities']
    channels = ['assembler', 'inserter', 'belt', 'pole',
                'direction', 'recipe', 'item', 'sourcesink']
    X = 1

    if trim_topleft:
        left, top = bound_bp_by_json(js)
        for entity in entities:
            entity['position']['x'] -= left
            entity['position']['y'] -= top

    mats = np.zeros((w, h, len(channels)), dtype=np.int16)

    # build the recipe‑id map once
    recipe_id = _make_recipe_index(entities)
    
    for entity in entities:
        name, width, height = entity['name'], 1, 1  # Default values.
        is_directional = False
        idx = get_item_id_from_name(name)
        if name.startswith("assembling-machine"):
            key = channels.index("assembler")
            width, height = 3, 3
        elif name in inserter_index:
            key = channels.index("inserter")
            is_directional = True
        elif name in belt_index:
            is_directional = True
            key = channels.index("belt")
            if 'splitter' in name:
                if entity.get('direction', 0) in [0, 4]:
                    width, height = 2, 1
                else:
                    width, height = 1, 2
        elif name in pole_index:
            if name == 'substation':
                width, height = 2, 2
            key = channels.index("pole")
        elif name == 'ee-infinity-loader':  # Source or sink.
            is_directional = True
            key = channels.index("belt")
            if entity.get('filter') is None:  # Sink.
                X = -1
            idx = get_item_id_from_name('underground-belt')  # Treated as underground belts.
        else:
            logging.info(f"Skipping {entity['name']}.")
            entity = None
            continue

        # Placement logic
        logging.info(f"Placing {entity['name']}.")
        # Determining the item's bounds.
        lx, ty = get_entity_topleft(entity)
        rx, by = lx + width, ty + height
        # Placing the building on the matrix.
        mats[lx:rx, ty:by, key] = X
        mats[lx:rx, ty:by, channels.index('item')] = idx
        
        logging.info(f"Taking up: {ty}, {lx} to {by}, {rx}.")
        if is_directional:
            mats[lx:rx, ty:by, channels.index("direction")] = entity.get('direction', 0) + 1

        # recipe
        if key == 0 and "recipe" in entity:
            mats[lx:rx, ty:by, channels.index("recipe")] = recipe_id[entity["recipe"]]

    # NOTE: center_in_N doesn't seem to work.
    if center:
        N = max(w, h)
        mats = center_in_N(matrix=mats, N=N)
    return mats


def _make_recipe_index(entities) -> dict[str, int]:
    """
    Return a dict mapping each recipe_name to a unique integer ID,
    numbered 0, 1, 2… in order of first appearance (ignores icons).
    """
    index: dict[str, int] = {}
    next_id = 1
    for e in entities:
        r = e.get("recipe")
        if r is None or r in index:
            continue
        index[r] = next_id
        next_id += 1
    return index


def json_to_manychannel_matrix(js: dict, w, h,
                            trim_topleft: bool=True,
                            center=False):
    opacity_channels = ['assembler', 'belt', 'inserter', 'pole']
    other_channels = ['direction', 'recipe', 'item', 'kind', 'sourcesink']
    entities = js['blueprint']['entities']

    if trim_topleft:
        left, top = bound_bp_by_json(js)
        for entity in entities:
            entity['position']['x'] -= left
            entity['position']['y'] -= top

    assemblers = [e for e in entities if e['name'].startswith('assembling-machine')]
    belts = [e for e in entities if e['name'] in belt_index]
    inserters = [e for e in entities if e['name'] in inserter_index]
    poles = [e for e in entities if e['name'] in pole_index]

    assembler_matrices = make_assembler_matrix(assemblers, w, h)
    belt_matrices = make_belt_matrix(belts, w, h)
    inserter_matrices = make_inserter_matrix(inserters, w, h)
    pole_matrices = make_pole_matrix(poles, w, h)
    submatrix_dict_list = [assembler_matrices, belt_matrices, inserter_matrices, pole_matrices]

    opacity_matrix_list = []
    for k, m in zip(opacity_channels, submatrix_dict_list):
        opacity_matrix_list.append(m[k])  # Adds the opacity channels in order.

    channel_matrices = {}
    for channel in other_channels:
        # First, find the maximum number of subchannel dimensions for this channel type
        max_channels = max(
            (m[channel].shape[2] for m in submatrix_dict_list if channel in m),
            default=0
        )
        
        if max_channels > 0:
            # Initialize the combined matrix with proper channel depth
            combined = np.zeros((w, h, max_channels), dtype=int)
            
            # Sum up corresponding channels from each submatrix
            for submatrix in submatrix_dict_list:
                if channel in submatrix:
                    # Get the current submatrix for this channel
                    curr_matrix = submatrix[channel]
                    # Add it to the combined matrix, but only up to the number of channels in curr_matrix
                    combined[:, :, :curr_matrix.shape[2]] += curr_matrix
            
            channel_matrices[channel] = combined

    mats = np.concatenate(opacity_matrix_list + list(channel_matrices.values()), axis=2)

    if center:
        N = max(w, h)
        mats = center_in_N(matrix=mats, N=N)

    return mats


# So, we'll actually go back to our roots here: A dictionary that maps the channel names
# to the matrices.
# And then we'll use that to merge stuff together.
# Note! These "channel names" actually correspond to MULTIPLE channels if they're one-hots.
def make_assembler_matrix(assemblers, w, h):
    """ Takes a blueprint JSON and makes the assembler matrix associated with it."""
    channels = ['assembler', 'recipe', 'item']
    recipe_id = _make_recipe_index(assemblers)

    opacity_matrix = np.zeros((w, h, 1), dtype=int)
    recipe_matrix = np.zeros((w, h, 5), dtype=int)  # Allows for 5 recipes at once.
    item_id_matrix = np.zeros((w, h, 3), dtype=int)  # Allows for up to 3 tiers of assemblers.
    for entity in assemblers:
        name = entity['name']
        idx = int(name[-1])-1  # Maps 1,2,3 to 0,1,2 for channels.

        # Determining the item's bounds.
        lx, ty = get_entity_topleft(entity)
        rx, by = lx+3, ty+3
        opacity_matrix[lx:rx, ty:by, 0] = 1
        item_id_matrix[lx:rx, ty:by, idx] = 1
        if 'recipe' in entity:
            rid = recipe_id[entity["recipe"]] - 1  # Maps 1,2,3,4,5 to 0,1,2,3,4.
            if rid > 4:
                logging.error("Too many recipes!!!")
            recipe_matrix[lx:rx, ty:by, rid] = 1

    map_matrix = {
        'assembler': opacity_matrix,
        'item': item_id_matrix,
        'recipe': recipe_matrix
    }
    
    return map_matrix

def make_belt_matrix(belts, w, h):
    channels = ['belt', 'direction', 'item', 'kind', 'sourcesink']
    # 1 for opacity, 4 channels for direction, 3 for item, 2 for sourcesink
    opacity_matrix = np.zeros((w, h, 1), dtype=int)
    direction_matrix = np.zeros((w, h, 4), dtype=int)
    item_id_matrix = np.zeros((w, h, 3), dtype=int)  # Allows for up to 3 tiers of belts.
    kind_matrix = np.zeros((w, h, 3), dtype=int)  # Allows for 3 kinds of belts.
    sourcesink_matrix = np.zeros((w, h, 2), dtype=int)
    for entity in belts:
        name = entity['name']
        idx = belt_index[name] - 1
        # CRUDE HACK
        direction = (entity.get('direction', 0)%8)//2
        # Set the kind.
        if name in underground_index:
            kind_idx = 1
        elif name in splitter_index:
            kind_idx = 2
        else:
            kind_idx = 0

        # Determining the item's bounds.
        lx, ty = get_entity_topleft(entity)
        if kind_idx < 2:  # Not a splitter.
            dx, dy = 1, 1
        elif direction in [0, 2]:
            # Wide.
            dx, dy = 2, 1
        elif direction in [1, 3]:
            dx, dy = 1, 2
        else:
            raise RepresentationError(f"Direction had value of {direction}.")

        rx, by = lx+dx, ty+dy
        opacity_matrix[lx:rx, ty:by, 0] = 1
        item_id_matrix[lx:rx, ty:by, idx] = 1
        kind_matrix[lx:rx, ty:by, kind_idx] = 1
        direction_matrix[lx:rx, ty:by, direction] = 1
        if name == 'ee-infinity-loader':
            # INITIALIZE CHAOS METRICS
            # If there's no filter, it's a sink.
            srcsink = 1 if entity.get('filter') is None else 0
            sourcesink_matrix[lx:rx, ty:by, srcsink] = 1

    map_matrix = {
        'belt': opacity_matrix,
        'item': item_id_matrix,
        'direction': direction_matrix,
        'kind': kind_matrix,
        'sourcesink': sourcesink_matrix
    }
    
    return map_matrix


def make_inserter_matrix(inserters, w, h):
    channels = ['inserter', 'direction', 'item']
    opacity_matrix = np.zeros((w, h, 1), dtype=int)
    direction_matrix = np.zeros((w, h, 4), dtype=int)
    item_id_matrix = np.zeros((w, h, 3), dtype=int)  # Allows for up to 3 kinds of belts.
    for entity in inserters:
        name = entity['name']
        idx = inserter_index[name] - 1
        # HACK!!! TODO: MAKE THIS NOT A HACK
        direction = (entity.get('direction', 0)%8)//2

        # Determining the item's bounds.
        lx, ty = get_entity_topleft(entity)
        rx, by = lx+1, ty+1
        opacity_matrix[lx:rx, ty:by, 0] = 1
        item_id_matrix[lx:rx, ty:by, idx] = 1
        direction_matrix[lx:rx, ty:by, direction] = 1

    map_matrix = {
        'inserter': opacity_matrix,
        'item': item_id_matrix,
        'direction': direction_matrix,
    }
    
    return map_matrix


def make_pole_matrix(poles, w, h):
    channels = ['pole', 'item']
    opacity_matrix = np.zeros((w, h, 1), dtype=int)
    item_id_matrix = np.zeros((w, h, 3), dtype=int)
    
    for entity in poles:
        name = entity['name']
        idx = pole_index[name] - 1
        lx, ty = get_entity_topleft(entity)
        pd = 1 if idx < 2 else 2  # Large poles and substations are 2x2.
        rx, by = lx+pd, ty+pd
        
        opacity_matrix[lx:rx, ty:by, 0] = 1
        item_id_matrix[lx:rx, ty:by, idx] = 1
    
    return {
        'pole': opacity_matrix,
        'item': item_id_matrix
    }

def split_matrix_into_entities(matrix):
    """Split a 21-channel matrix into entity-specific matrices.
    
    The input matrix has channels in this order:
    - opacity: [assembler, belt, inserter, pole] (4)
    - direction: [N, E, S, W] (4) 
    - recipe: [r1, r2, r3, r4, r5] (5)
    - item: [i1, i2, i3] (3)
    - kind: [k1, k2, k3] (3)
    - sourcesink: [source, sink] (2)
    
    Returns:
    - Dictionary with keys 'assembler', 'belt', 'inserter', 'pole'
      Each value is a binary matrix where 1 indicates that type of entity
    """
    h, w = matrix.shape[:2]
    
    # Create entity matrices
    assembler_matrix = np.zeros((h, w), dtype=int)
    belt_matrix = np.zeros((h, w), dtype=int)
    inserter_matrix = np.zeros((h, w), dtype=int)
    pole_matrix = np.zeros((h, w), dtype=int)
    
    # Extract opacity channels (first 4 channels)
    assembler_mask = matrix[:, :, 0]
    belt_mask = matrix[:, :, 1]
    inserter_mask = matrix[:, :, 2]
    pole_mask = matrix[:, :, 3]
    
    # For each position where an entity exists (opacity=1),
    # assign a unique integer based on the other channels
    
    # Assemblers: influenced by recipe (channels 8-12)
    recipe_channels = matrix[:, :, 8:13]
    recipe_indices = np.argmax(recipe_channels, axis=2) + 1
    assembler_matrix[assembler_mask == 1] = recipe_indices[assembler_mask == 1]
    
    # Belts: influenced by direction (4-7) and kind (16-18)
    direction_channels = matrix[:, :, 4:8]
    kind_channels = matrix[:, :, 16:19]
    direction_indices = np.argmax(direction_channels, axis=2) + 1
    kind_indices = np.argmax(kind_channels, axis=2) + 1
    belt_matrix[belt_mask == 1] = direction_indices[belt_mask == 1] + (kind_indices[belt_mask == 1] - 1) * 4
    
    # Inserters: influenced by direction (4-7)
    inserter_matrix[inserter_mask == 1] = direction_indices[inserter_mask == 1]
    
    # Poles: just use the mask since they don't have orientation
    pole_matrix[pole_mask == 1] = 1
    
    return {
        'assembler': assembler_matrix,
        'belt': belt_matrix,
        'inserter': inserter_matrix,
        'pole': pole_matrix
    }