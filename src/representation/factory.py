""" factory.py
Handles the Factory class and various methods related to it.
"""
import logging
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional

from draftsman.blueprintable import Blueprint, get_blueprintable_from_JSON, get_blueprintable_from_string
from draftsman.utils import string_to_JSON

from .utils import recursive_blueprint_book_parse, RepresentationError
from .matrix import (blueprint_to_opacity_matrices,
                     json_to_6channel_matrix,
                     json_to_7channel_matrix,
                     json_to_8channel_matrix,
                     json_to_manychannel_matrix)
# You surely will not regret putting a global variable in your representation module.
REPR_VERSION = 5


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
        # print(j['blueprint'].keys())
        if 'blueprint' not in j:  # not a proper blueprint
            if 'blueprint-book' in j:  # this is a BOOK
                raise RepresentationError("Attempted to represent a blueprint book as a Factory.")
            else:
                logging.debug(j)
                raise RepresentationError("Failed for an unknown issue.")
        # if 'wires' in j['blueprint']:
        #     del(j['blueprint']['wires'])
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
                repr_version: int = REPR_VERSION,
                **kwargs):
        if repr_version == 1:
            mats = blueprint_to_opacity_matrices(self.blueprint, *dims, **kwargs)
        elif repr_version == 2:
            mats = json_to_6channel_matrix(self.json, *dims, **kwargs)
        elif repr_version == 3:
            mats = json_to_7channel_matrix(self.json, *dims, **kwargs)
        elif repr_version == 4:
            mats = json_to_8channel_matrix(self.json, *dims, **kwargs)
        elif repr_version == 5:
            mats = json_to_manychannel_matrix(self.json, *dims, **kwargs)
        else:
            raise RepresentationError(f"Unknown repr_version {repr_version}")
        return mats

    def get_tensor(self, dims: tuple[int, int]):
        # Uses default.
        import torch
        mats = self.get_matrix(dims).astype(np.float32)
        return torch.from_numpy(mats).unsqueeze(0).permute(0, 3, 1, 2)
