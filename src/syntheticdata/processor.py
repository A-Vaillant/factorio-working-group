"""
processor.py


"""
import json, random
import numpy as np
from pathlib import Path
from copy import deepcopy

from draftsman.blueprintable import Blueprint, Blueprintable, get_blueprintable_from_string

from src.representation import blueprint_to_opacity_matrices, recursive_blueprint_book_parse, map_entity_to_key
from src.machinelearning import center_in_15x15

data_root = Path('data')


class FactoryLoader():
    def __init__(self, raw_data_src,
                 update_direction: bool=True):
        # We're gonna use raw/txt/av here.
        loading_root = data_root / raw_data_src
        self.factories = dict()
        if 'txt' in str(raw_data_src):  # Text loader.
            manifile = loading_root / Path('manifest.json')
            if not manifile.exists():
                raise FileExistsError("SCREAMING!!!!!!!!!!  ")
            
            with manifile.open() as mf:
                manidata = json.load(mf)
            for k, v in manidata['data_files'].items():
                with (loading_root / v).open() as bfile:
                    v = get_blueprintable_from_string(bfile.read())
                if isinstance(v, Blueprint):
                    self.factories[k] = v
                    continue
                vs = recursive_blueprint_book_parse(v)
                for ix, v in enumerate(vs):
                    self.factories[f"{k}-{ix}"] = v
            if update_direction:  # For 1.0 compat.
                for factory in iter(self):
                    for e in factory.entities:
                        try:
                            e.direction *= 2
                        except AttributeError:
                            pass
        elif 'csv' in str(raw_data_src):  # csv loader
            pass
        elif 'json' in str(raw_data_src):  # json loader
            pass
        else:
            raise FileNotFoundError("Only works for blueprint packages.")
        
        


    def __iter__(self):
        # Lets you iterate through the Blueprints by just iterating over the Loader.
        return iter(self.factories.values())
    
    def random_sample(self):
        return random.choice(list(self.factories.values()))
    

class EntityPuncher():
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
        for ix, entity in enumerate(self.factory.entities):
            # print(entity)
            channel_name = map_entity_to_key(entity)
            if not channel_name:
                continue
            if channel_name == 'pole' and ignore_electricity:
                continue
            factory_copy = deepcopy(self.factory)
            factory_copy.entities.pop(ix)
            root_sequence.append([factory_copy,
                                  self.channels.index(channel_name),
                                  entity.tile_position])
        return root_sequence
    

# In the future, we'll be able to load from a processed dataset
# which will have a lot of the stuff already stored.
datasets = {
    'av-redscience': 'raw/txt/av'
}

def load_dataset(dataset_name: str='av-redscience',
                  **kwargs):
    """ dataset_name: The name of a prepared dataset. 

    Right now, only does the redscience entity holepunch dataset.
        Returns 4 arrays.
        Form: (4-channel grid, channel) -> (4-channel grid, x,y coordinate of new entity)
    """
    fl = FactoryLoader(datasets[dataset_name], **kwargs)
    Xs = []
    y = []
    y_pos = []
    indices = []
    for k, v in fl.factories.items():
        ep = EntityPuncher(v)
        rs = ep.get_removal_sequences(ignore_electricity=True)
        try:
            v_mat = center_in_15x15(blueprint_to_opacity_matrices(v))
        except ValueError:
            continue
        # print(len(rs))
        for (holepunched, ix, pos) in rs:
            # Map the factory to a matrix, then organize.
            hp_mat = center_in_15x15(blueprint_to_opacity_matrices(holepunched))
            Xs.append(hp_mat)
            y.append(v_mat)
            y_pos.append(pos._data)
            indices.append(ix)
            # np.append(Xs, hp_mat)
            # np.append(y, v_mat)
            # np.append(y_pos, np.array(pos._data))
            # np.append(indices, ix)
    return np.array(Xs), y, y_pos, np.array(indices)
