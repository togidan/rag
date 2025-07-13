import mmh3
from typing import Set


def generate_text_hash(text: str, attempt: int = 0) -> int:
    """
    Generate a consistent INT64 hash for text using MurmurHash3.
    
    Args:
        text: Text to hash
        attempt: Attempt number for collision handling
        
    Returns:
        64-bit integer hash value
    """
    seed = 42 + attempt
    hash_32 = mmh3.hash(text, seed=seed)
    hash_64 = abs(hash_32) % (2**63 - 1)
    return hash_64


def generate_unique_hash(text: str, existing_hashes: Set[int]) -> int:
    """
    Generate a unique hash, handling collisions by trying different seeds.
    
    Args:
        text: Text to hash
        existing_hashes: Set of existing hash values to avoid
        
    Returns:
        Unique hash value
        
    Raises:
        ValueError: If unable to generate unique hash after max attempts
    """
    attempt = 0
    max_attempts = 100
    
    while attempt < max_attempts:
        hash_value = generate_text_hash(text, attempt)
        if hash_value not in existing_hashes:
            return hash_value
        attempt += 1
    
    raise ValueError(f"Could not generate unique hash for text after {max_attempts} attempts")