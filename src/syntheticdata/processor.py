"""
processor.py


"""
import json, random
import numpy as np
import logging
from pathlib import Path
from copy import deepcopy

from src.pipeline import FactoryLoader
from src.representation import blueprint_to_opacity_matrices, map_entity_to_key, center_in_N, Factory


def make_rotations(matrix, hwc=True):
    # Assumption: Matrix is in HWC form.
    rotations = []
    seen = set()

    # hwc is numpy default, chw is torch tensor default
    if hwc:
        axes = (0, 1)
    else:
        axes = (1, 2)
    for i in range(1, 4):
        rotated = np.rot90(matrix, k=i, axes=axes)  # Rotate on (H, W)
        key = rotated.tobytes()   # A funny caching thing.
        if key not in seen:
            seen.add(key)
            rotations.append(rotated)
        else:
            logging.info(f"Rotation {i * 90}° is not unique, skipping.")

    return rotations


def make_translations(matrix, count=5):
    c, h, w = matrix.shape  # NOTE: This assumes CxHxW, which is not what we use by default!
    # We probably don't need to use this, though, since CNNs are already good at translation invariance.
    pad = 2
    padded = np.pad(matrix, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    translations = []
    seen = set()

    for dx in range(-pad, pad + 1):
        for dy in range(-pad, pad + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(padded)
            shifted[:, pad+dy:pad+dy+h, pad+dx:pad+dx+w] = matrix
            key = shifted.tobytes()
            if key not in seen:
                seen.add(key)
                translations.append(shifted)
            if len(translations) >= count:
                break
        if len(translations) >= count:
            break

    if len(translations) < count:
        logging.info(f"Only generated {len(translations)} unique translations out of requested {count}.")


class EntityPuncher():
    # Transforms.
    channels = ('assembler', 'inserter', 'belt', 'pole')

    def __init__(self, factory):
        self.factory = factory

    def get_removal_sequences(self,
                              root_sequence=None,
                              ignore_electricity: bool=False):
        """ For the factory, returns all variants where an entity was removed
        as their matrix representations. """
        if root_sequence is None:
            root_sequence = []

        # levels -= 1
        for ix, entity in enumerate(self.factory.blueprint.entities):
            # print(entity)
            channel_name = map_entity_to_key(entity)
            if not channel_name:
                continue
            if channel_name == 'pole' and ignore_electricity:
                continue
            factory_copy = deepcopy(self.factory.blueprint)
            factory_copy.entities.pop(ix)
            root_sequence.append([factory_copy,
                                  self.channels.index(channel_name),
                                  entity.tile_position])
        return root_sequence
    

# TODO: Move this.
datasets = {
    'av-redscience': 'raw/txt/av',
    'factorio-tech-json': 'raw/json/factorio-tech',
    'factorio-tech': 'raw/csv/factorio-tech',
    'factorio-codex': 'raw/csv/factorio-codex',
    'idan': 'raw/csv/idan_blueprints.csv',
}

def load_dataset(dataset_name: str='av-redscience',
                  **kwargs):
    """ dataset_name: The name of a prepared dataset. 
    """
    return FactoryLoader(datasets[dataset_name])
    
def load_training_data(dataset_name, **kwargs):
    fl = FactoryLoader(datasets[dataset_name], **kwargs)
    Xs = []
    y = []
    indices = []
    for k, v in fl.factories.items():
        ep = EntityPuncher(v)
        rs = ep.get_removal_sequences(ignore_electricity=True)
        try:
            v_mat = center_in_N(blueprint_to_opacity_matrices(v), N=20)
        except ValueError:
            continue
        # print(len(rs))
        for (holepunched, ix, pos) in rs:
            # Map the factory to a matrix, then organize.
            hp_mat = center_in_N(blueprint_to_opacity_matrices(holepunched), N=20)
            Xs.append(hp_mat)
            y.append(v_mat)
            indices.append(ix)
    return np.array(Xs), y, np.array(indices)
