"""
utils.py

Just some extra stuff.
"""
import hashlib


def calculate_file_hash(filename: str) -> str:
    """Calculate MD5 hash of file for idempotency checking."""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def hash_bp_row(bp_string):
    """Hash a blueprint string to generate a unique filename."""
    return hashlib.md5(bp_string.encode()).hexdigest()[:10]

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


def visualize_multichannel(matrix_dict):
    for k, v in matrix_dict.items():
        print(f"channel {k}")
        print(v.T)
