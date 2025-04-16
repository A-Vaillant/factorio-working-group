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

