"""
processor.py
"""
import json, random
import numpy as np
import logging
from pathlib import Path
import itertools
from copy import deepcopy

from draftsman.blueprintable import get_blueprintable_from_string
from draftsman.constants import Direction
from src.representation import (Factory,
                                map_entity_to_key)


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def copy_factory(factory):
    # Used to avoid a deepcopy bug.
    bp = get_blueprintable_from_string(factory.blueprint.to_string())
    return Factory.from_blueprint(bp)

class PuncherPrototype():
    # Base class for a PoD-based dataset synthesizer.
    # Just implement the order you destroy and the entities you don't destroy and "ur set".
    channels = ['assembler', 'inserter', 'belt', 'pole',
                'direction', 'recipe', 'item', 'sourcesink']
    
    def __init__(self, factory):
        # We keep updating the factory object held by the puncher.
        self.original_factory = factory
        factory.blueprint.wires = []  # Clear associations, as we don't use them.
        self.blueprint = deepcopy(factory.blueprint)
        self.removed_positions = set()
        logger.debug(f"Initialized EntityPuncher with {len(self.blueprint.entities)} entities")
        
    def _save_entities(self):
        # Adds entities to self.removed_positions to prevent them from being deleted.
        raise NotImplemented()
    
    def _next_removal_order(self):
        raise NotImplemented()
    
    def generate_state_action_pairs(self, num_pairs: int=None):
        """
        Returns lists of (before_factory, after_factory, repair_action_idx).
        """
        pairs = []
        # Add identity transforms.
        factory = Factory.from_blueprint(deepcopy(self.blueprint))
        pairs.append((factory, factory, 0))

        self._save_entities()
        entity_iterator =  self._next_removal_order(self.blueprint.entities)
        if num_pairs is not None:
            entity_iterator = itertools.islice(entity_iterator, 0, num_pairs)
        for removable in entity_iterator:
            ch = self.channels.index(map_entity_to_key(removable))

            # record 'before'
            before_factory = Factory.from_blueprint(deepcopy(self.blueprint))
            repair_action = ch

            self.removed_positions.add(tuple(removable.tile_position._data))
            self.blueprint.entities.recursive_remove(removable)

            after_factory = Factory.from_blueprint(deepcopy(self.blueprint))

            pairs.append((before_factory, after_factory, repair_action))

        # unpack into arrays
        X_before = np.stack([p[0] for p in pairs], axis=0)
        X_after  = np.stack([p[1] for p in pairs], axis=0)
        y        = np.array([p[2] for p in pairs])
        return X_before, X_after, y
    

class SeedPuncher(PuncherPrototype):
    def _save_entity(self, entity):
        self.removed_positions.add(tuple(entity.tile_position._data))
        
    def _save_entities(self):
        from collections import defaultdict
        assemblers = [e for e in self.original_factory.blueprint.entities if map_entity_to_key(e)=='assembler']
        inserters_to = defaultdict(list)
        inserters_from = defaultdict(list)
        for i, asm in enumerate(assemblers):
            asm_x, asm_y = asm.tile_position._data
            is_in_assembler = lambda x, y: (asm_x <= x <= asm_x+2) and (asm_y <= y <= asm_y+2)
            for ins in self.original_factory.blueprint.entities:
                ins.direction = Direction(ins.direction)
                # TODO: An additional case for long-armed inserters.
                if map_entity_to_key(ins) != 'inserter': continue  # Skip non-inserters.
                if ins['name'].contains('long'):
                    magnitude = 2
                else:
                    magnitude = 1
                dropoff = ins.tile_position + ins.direction.to_vector(magnitude=magnitude)
                pickup = ins.tile_position - ins.direction.to_vector(magnitude=magnitude)
                
                if is_in_assembler(dropoff):
                    inserters_to[i].append(ins)
                elif is_in_assembler(pickup):
                    inserters_from[i].append(ins)
                else:
                    pass
            
        for ix, assembler in enumerate(assemblers):
            self._save_entity(assembler)
            for ins in inserters_to[ix]:
                self._save_entity(ins)
                        
    def _next_removal_order(self):
        # TODO:
        # First, we'll place the belts by the adjacent inserters that are there.
        # Then we'll remove stuff randomly and in a fashion similar to a garden of forking branches, which will produce
        # some large permutation of data.
        # (If there's n entities to remove, that's n! as the upper bound.)
        ...
        

        
class EntityPuncher(PuncherPrototype):
    def _save_entities(self):
        assemblers = [e for e in self.original_factory.blueprint.entities if map_entity_to_key(e)=='assembler']
        # pick one assembler at random
        chosen = random.choice(assemblers)
        position = tuple(chosen.tile_position._data)
        self.removed_positions.add(position)
        logger.debug(f"In _set_final_blueprint, marking position {position} as removed")
        # Simply chooses a random assembler and pretends like we already removed it. Good job!

    def _sort_assemblers(self, entities):
        assemblers = [e for e in entities if map_entity_to_key(e)=='assembler']
        logger.debug(f"Sorting {len(assemblers)} assemblers")
        return sorted(
            assemblers,
            key=lambda e: getattr(e, 'recipe', getattr(e, 'entity_number', 0))
        )

    def _belt_chains(self, entities):
        chains = []
        unused = set([id(e) for e in entities if map_entity_to_key(e) == 'belt'])
        pos_map = {tuple(e.tile_position): e for e in entities if map_entity_to_key(e) == 'belt'}

        # Correct 4-way direction mapping
        direction_to_vector = {
            0: (0, -1),  # Up (north)
            4: (1, 0),   # Right (east)
            8: (0, 1),   # Down (south)
            12: (-1, 0),  # Left (west)
        }

        for e in entities:
            if map_entity_to_key(e) != 'belt' or id(e) not in unused:
                continue

            chain = [e]
            unused.remove(id(e))
            cur = e

            while True:
                direction_vector = direction_to_vector.get(cur.direction, (0, 0))
                nbr_pos = (
                    cur.tile_position[0] + direction_vector[0],
                    cur.tile_position[1] + direction_vector[1]
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


    def _inserters_ready(self, entities):
        return sorted(
            [e for e in entities if map_entity_to_key(e)=='inserter'],
            key=lambda e: getattr(e, 'entity_number', 0)
        )

    def _poles_random(self, entities):
        poles = [e for e in entities if map_entity_to_key(e)=='pole']
        random.shuffle(poles)
        return poles

    def _next_removal_order(self, entities):
        logger.debug(f"Starting _next_removal_order with {len(entities)} entities")
        logger.debug(f"Currently {len(self.removed_positions)} positions in removed_positions set: {self.removed_positions}")

        # yield assemblers, then belts, then inserters, then poles
        for e in itertools.chain(self._sort_assemblers(entities),
                                 self._belt_chains(entities),
                                 self._inserters_ready(entities),
                                 self._poles_random(entities)):
            # Note; why can't I just check entity inclusion directly...?
            position = tuple(e.tile_position._data)
            logger.debug(f"Considering entity at position {position}, type: {map_entity_to_key(e)}")
            if position not in self.removed_positions:
                logger.debug(f"Yielding entity at position {position} for removal")
                yield e
            else:
                logger.debug(f"SKIPPING entity at position {position} - already in removed_positions")
                continue