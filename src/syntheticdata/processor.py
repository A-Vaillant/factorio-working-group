"""
processor.py


"""
import json, random
import numpy as np
import logging
from pathlib import Path
from copy import deepcopy

from src.pipeline import FactoryLoader
from src.pipeline.datasets import datasets
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


# class EntityPuncher():
#     # Transforms.
#     channels = ('assembler', 'inserter', 'belt', 'pole')

#     def __init__(self, factory):
#         self.factory = factory

#     def get_removal_sequences(self,
#                               root_sequence=None,
#                               ignore_electricity: bool=False):
#         """ For the factory, returns all variants where an entity was removed
#         as their matrix representations. """
#         if root_sequence is None:
#             root_sequence = []

#         # levels -= 1
#         for ix, entity in enumerate(self.factory.blueprint.entities):
#             # print(entity)
#             channel_name = map_entity_to_key(entity)
#             if not channel_name:
#                 continue
#             if channel_name == 'pole' and ignore_electricity:
#                 continue
#             factory_copy = deepcopy(self.factory.blueprint)
#             factory_copy.entities.pop(ix)
#             root_sequence.append([factory_copy,
#                                   self.channels.index(channel_name),
#                                   entity.tile_position])
#         return root_sequence
    

class EntityPuncher():
    channels = ('assembler', 'inserter', 'belt', 'pole')

    def __init__(self, factory, N=20):
        self.blueprint = factory.blueprint
        self.N = N

    def _starting_blueprint(self):
        """
        Your 'Starting Set':
        - One random assembler
        - Plus maybe a few other random entities
        """
        bp = deepcopy(self.blueprint)
        assemblers = [e for e in bp.entities if map_entity_to_key(e)=='assembler']
        # pick one assembler at random
        chosen = random.choice(assemblers)
        bp.entities = [chosen] + random.sample(
            [e for e in bp.entities if e!=chosen], 
            k=min(3, len(bp.entities)-1)
        )
        return bp

    def is_valid_start_state(self, blueprint, max_total_entities=4):
        """Returns True if exactly 1 assembler and only a few total entities exist."""
        assembler_count = 0
        total_count = 0
        for e in blueprint.entities:
            if self.map_entity_to_key(e) is None:
                continue
            total_count += 1
            if self.map_entity_to_key(e) == 'assembler':
                assembler_count += 1
        return assembler_count == 1 and total_count <= max_total_entities

    def _sort_assemblers(self, entities):
        return sorted(
            [e for e in entities if map_entity_to_key(e)=='assembler'],
            key=lambda e: getattr(e, 'recipe', getattr(e, 'entity_number', 0))
        )

    def _belt_chains(self, entities):
        # naive: group belts by connected chains, reverse each chain
        from collections import defaultdict
        chains = []
        unused = set([id(e) for e in entities if map_entity_to_key(e)=='belt'])
        # print("\nUNUSED:\n" + str(unused))
        # for e in entities:
        #     print(str(e) + "\n\t" + str(e.direction) )
        pos_map = {tuple(e.tile_position): e for e in entities if map_entity_to_key(e)=='belt'}
        # print("\nPOS MAP: \n" + str(pos_map))
        # print("\nPOS MAP: \n" + str(pos_map.get((1,0))))
        for e in entities:
            if map_entity_to_key(e)!='belt' or id(e) not in unused: continue
            chain = [e]
            unused.remove(id(e))
            # walk one direction until no neighbor
            cur = e
            while True:
                nbr_pos = (
                    cur.tile_position[0] + cur.direction.vector[0], # x pos?
                    cur.tile_position[1] + cur.direction.vector[1]  # y pos?
                )
                nbr = pos_map.get(tuple(nbr_pos), None)
                if nbr and id(nbr) in unused:
                    chain.append(nbr)
                    unused.remove(id(nbr))
                    cur = nbr
                else:
                    break
            chains.append(list(reversed(chain)))
        return [e for chain in chains for e in chain]

    def _inserters_ready(self, entities, removed_ids):
        ready = []
        for e in entities:
            if map_entity_to_key(e)!='inserter': continue
            # if any neighbor (pos ±1) has been removed
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                p = (e.tile_position[0]+dx, e.tile_position[1]+dy)
                if p in removed_ids:
                    ready.append(e)
                    break
        return ready

    def _poles_random(self, entities):
        poles = [e for e in entities if map_entity_to_key(e)=='pole']
        random.shuffle(poles)
        return poles

    def _next_removal_order(self, entities, removed_ids):
        # yield assemblers, then belts, then inserters, then poles
        for e in self._sort_assemblers(entities):
            if id(e) not in removed_ids: yield e
        for e in self._belt_chains(entities):
            if id(e) not in removed_ids: yield e
        for e in self._inserters_ready(entities, removed_ids):
            if id(e) not in removed_ids: yield e
        for e in self._poles_random(entities):
            if id(e) not in removed_ids: yield e

    def generate_state_action_pairs(self, num_pairs: int):
        """
        Implements steps 1–6 from the paper.
        Returns lists of (before_mat, after_mat, repair_action_idx).
        """
        pairs = []
        # 1) Start from a random 'starting blueprint'
        # TODO we should start from an initial blueprint and end at a starting state
        current_bp = self._starting_blueprint()
        # current_bp = self.blueprint
        removed_ids = set()

        while len(pairs) < num_pairs:
            entities = current_bp.entities
            # 2) get next entity to remove in 'Path of Destruction' order
            try:
                to_remove = next(self._next_removal_order(entities, removed_ids))
            except StopIteration:
                # no more possible removals → restart from a fresh starting point
                current_bp = self._starting_blueprint()
                removed_ids.clear()
                continue

            # find its channel idx
            ch = self.channels.index(map_entity_to_key(to_remove))
            pos = to_remove.tile_position

            # 3) record 'before'
            before_mat = center_in_N(blueprint_to_opacity_matrices(current_bp), N=self.N)
            repair_action = ch

            # 4) apply removal
            bp_after = deepcopy(current_bp)
            for i,e in enumerate(bp_after.entities):
                if e.tile_position == pos and map_entity_to_key(e)==self.channels[ch]:
                    bp_after.entities.pop(i)
                    break
            removed_ids.add(tuple(pos))  # mark this pos as 'removed' for inserter logic

            # 5) check your noise/starting distribution—here we simply accept all
            #    (you can plug in your own predicate, e.g. count_entities(bp_after)>k)

            # 6) record the pair
            after_mat = center_in_N(blueprint_to_opacity_matrices(bp_after), N=self.N)
            pairs.append((before_mat, after_mat, repair_action))

            # move on—continue destroying on top of this state
            current_bp = bp_after

        # unpack into arrays
        X_before = np.stack([p[0] for p in pairs], axis=0)
        X_after  = np.stack([p[1] for p in pairs], axis=0)
        y        = np.array([p[2] for p in pairs])
        return X_before, X_after, y

    
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
