""" intake.py
Written hastily and without internet access, so likely doesn't work!
TODO: Import Draftsman, make sure CsvReading works, add in JsonOutputting.
"""
from pathlib import Path
import csv, json


# TODO Make this better.
source_directory = Path("../data/raw")
output_directory = Path("../data/processed")


def process_csvs(src_dir):
    blueprint_data = []
    for csv_file in src_dir.glob("*.csv"):
        # Assumption: We'll get a list of dicts.
        csv_data = csv.CsvReader(csv_file)
        blueprint_data += [parse_csv_row(row) for row in csv_data]
    # We're just gonna flatten the file list.
    # So now it's lists of blueprints.
    
def hash_bp_row(bp_string):
    # TODO: Use a hash function to figure out the filename.
    return ''

def recursive_blueprint_book_parse(bp_book: Blueprintable) -> list[Blueprint]:
    # Reached a leaf.
    if isinstance(Blueprint, bp_book):
        return [bp_book]

    blueprints = []
    for bp_node in bp_book.blueprints:
        blueprints += recursive_blueprint_book_parse(bp_node)
    return blueprints


# Output will be a list of dictionaries.
# Uhh, I guess we can just hash the blueprint string to come up with json filenames.
# And then use that to avoid doing extra work.
def parse_csv_row(row: dict,
                  filter_func=lambda x: True) -> list[dict]:
    bp_string = row['data']
    blueprintable = import_from_bp_string(bp_string)
    # Process the blueprint.
    blueprints: list = recursive_blueprint_book_parse(blueprintable)

    # TODO: Filter out blueprints that are "trivial".
    outputs = []
    for ix, blueprint in enumerate(blueprints):
        # Here's where we can add some augmentation data.
        expanded_row = {k:v for k,v in row.items()}
        expanded_row['data'] = blueprint.to_bp_string() 

        expanded_row['entities'] = quantify_entities(blueprint)
        expanded_row['tech-level'] = quantify_tech_level(blueprint)
        if not filter_func(expanded_row):
            continue

        # Some niceties.
        # TODO: Deal with null names.
        # TODO: Figure out blueprint titles, names, etc.
        bp_name = blueprint.title
        if not bp_name:
            # zero pad for coolness
            bp_name = f"{ix:000}"
        # Different names for blueprints in a blueprint book.
        expanded_row['name'] += f' - {bp_name}'
        outputs.append(expanded_row)
    return outputs


def quantify_entities(bp: Blueprint,
                      granularity: int=5) -> dict[str, int]:
    # granularity: How specific we want. 5 will be everything, lower ones
    # will group together entities into classes (conveyor belts, assemblers)
    # returns a Counter dict
    from collections import Counter
    # TODO: Think about other granularities.
    if granularity == 5:
        grain = lambda x: x

    # TODO make sure this is how you access the name
    c = Counter(grain(entity.name) for entity in bp.entities)
    return c


# TODO: Write a map from every Factorio entity to the science tier.
tech_map = dict()
def quantify_tech_level(bp: Blueprint) -> int:
    # Using Factorio 1.0. 0 is initial, 1 is red science.
    tech_levels = {e: tech_map(e.name) for e in bp.entities}
    return max(tech_levels.values())


with __name__ == "__main__":
