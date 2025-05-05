import csv, json, sys
import random
import logging
import itertools

import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path

from draftsman.blueprintable import Blueprintable, get_blueprintable_from_string
from draftsman.error import MalformedBlueprintStringError
from draftsman.utils import string_to_JSON

from src.representation import Factory, recursive_json_parse, recursive_blueprint_book_parse

# TODO: Move this to some kind of configuration.
data_root = Path('data')


class FactoryLoader():
    def __init__(self, raw_data_src,
                 update_direction: bool=False,
                 base_path=data_root,
                 use_json: bool=True):
        # We're gonna use raw/txt/av here.
        loading_root = (base_path / 'raw') / raw_data_src
        self.factories = dict()
        if 'txt' in str(raw_data_src):  # Text loader.
            manifile = loading_root / Path('manifest.json')
            if not manifile.exists():
                raise FileExistsError("SCREAMING!!!!!!!!!!  ")
            
            with manifile.open() as mf:
                manidata = json.load(mf)
            for k, v in manidata['data_files'].items():
                with (loading_root / v).open() as bfile:
                    bpstring = bfile.read()
                    if use_json:
                        v = string_to_JSON(bpstring)
                        self.update_factories_via_json(k, v)
                    else:
                        v = get_blueprintable_from_string(bpstring)
                        self.update_factories_via_blueprintable(k, v)

        elif 'csv' in str(raw_data_src):  # csv loader
            csv.field_size_limit(sys.maxsize)
            # Allows us to load singular CSVs.
            if loading_root.suffix == '.csv':
                iteration_range = [loading_root]
            else:
                iteration_range = loading_root.glob("*.csv")
            for csvfile in iteration_range:
                with csvfile.open(newline='') as cf:
                    reader = csv.DictReader(cf)
                    # Convert each row to JSON format using the same string_to_JSON function
                    try:
                        for row in reader:
                            k = row.get('name') or csvfile.name[:25]
                            try:
                                v = string_to_JSON(row['data'])
                            except MalformedBlueprintStringError:
                                logging.warning(f"Malformed blueprint string detected for {k}. Skipping.")
                            self.update_factories(k, v)
                    except UnicodeDecodeError:
                        logging.warning(f"Could not use {str(csvfile)} due to UnicodeDecodeError.")

        elif 'json' in str(raw_data_src):  # json loader
            for jsonfile in loading_root.glob("*.json"):
                k = jsonfile.name[:25]  # truncate the name for no reason
                with jsonfile.open() as jf:
                    info = json.load(jf)
                    v = string_to_JSON(info['data'])
                self.update_factories(k, v)
        else:
            raise FileNotFoundError("Only works for blueprint packages.")
        
        # if update_direction:  # For 1.0 compat.
        #     for factory in iter(self):
        #         for e in factory.entities:
        #             try:
        #                 e.direction *= 2
        #             except AttributeError:
        #                 pass
   
    def update_factories_via_json(self, k: str,
                         v: dict):
        vs = []
        vs += recursive_json_parse(v)
        if len(vs) > 1:
            for ix, v in enumerate(vs):
                self.factories[f"{k}-{ix}"] = Factory(json=v)
        else:
            self.factories[k] = Factory(json=v)
    
    def update_factories_via_blueprintable(self, k: str,
                                           v: Blueprintable):
        vs = []
        vs += recursive_blueprint_book_parse(v)
        if len(vs) > 1:
            for ix, v in enumerate(vs):
                self.factories[f"{k}-{ix}"] = Factory.from_blueprint(v)
        else:
            self.factories[k] = Factory.from_blueprint(v)

    def __iter__(self):
        # Lets you iterate through the Blueprints by just iterating over the Loader.
        return iter(self.factories.values())
    
    def random_sample(self):
        return random.choice(list(self.factories.values()))


# This should probably not bother with the filtering and just focus on having
# Factories which have matrices inside them, no?
class MatrixDataset(IterableDataset):
    # An iterable dataset
    def __init__(self, iterable_obj: FactoryLoader,
                 repr_version=3, center=False,
                 filters=None,
                 num_samples=None,
                 N=20):
        self.io = iterable_obj
        self.version = repr_version
        self.center = center
        self.dims = (N,N)
        if filters is None:
            filters = []
        self.filters = filters
        self.num_samples = num_samples

    def __iter__(self):
        # Includes filtering early on.
        io = self.io
        for X in self.filters:
            io = filter(X, io)
        out = iter((f, f.get_matrix(dims=self.dims,
                                 repr_version=self.version,
                                 center=self.center,
                                 )) for f in io)
        if self.num_samples is not None:
            out = itertools.islice(out, 0, self.num_samples)
        return out


def collate_numpy_matrices(batch):
    levels, conditions, actions = zip(*batch)
    
    # Stack levels (convert to tensor and change to CHW format)
    levels_np = np.stack(levels)  # Shape: (batch_size, 20, 20, 7)
    levels_tensor = torch.from_numpy(levels_np).permute(0, 3, 1, 2)  # Shape: (batch_size, 7, 20, 20)
    
    # Handle conditions (may be None)
    valid_conditions = [c for c in conditions if c is not None]
    if valid_conditions:
        conditions_tensor = torch.tensor(valid_conditions, dtype=torch.float32)
        has_condition = torch.ones(len(conditions), dtype=torch.bool)
        for i, c in enumerate(conditions):
            has_condition[i] = c is not None
    else:
        conditions_tensor = None
        has_condition = torch.zeros(len(conditions), dtype=torch.bool)
    
    # Stack actions
    actions_np = np.stack(actions)  # Shape: (batch_size, 20, 20, 7)
    actions_tensor = torch.from_numpy(actions_np).permute(0, 3, 1, 2)  # Shape: (batch_size, 7, 20, 20)
    
    return levels_tensor, (conditions_tensor, has_condition), actions_tensor
