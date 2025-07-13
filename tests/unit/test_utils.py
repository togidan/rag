import pytest

from app.utils.hash_utils import generate_text_hash, generate_unique_hash


@pytest.mark.unit
class TestHashUtils:
    """Unit tests for hash utility functions."""
    
    def test_generate_text_hash_consistency(self):
        """Test that the same text generates the same hash."""
        text = "test document content"
        hash1 = generate_text_hash(text)
        hash2 = generate_text_hash(text)
        
        assert hash1 == hash2
        assert isinstance(hash1, int)
        assert hash1 > 0
    
    def test_generate_text_hash_different_attempts(self):
        """Test that different attempts generate different hashes."""
        text = "test document content"
        hash1 = generate_text_hash(text, attempt=0)
        hash2 = generate_text_hash(text, attempt=1)
        
        assert hash1 != hash2
    
    def test_generate_text_hash_different_text(self):
        """Test that different texts generate different hashes."""
        hash1 = generate_text_hash("text one")
        hash2 = generate_text_hash("text two")
        
        assert hash1 != hash2
    
    def test_generate_unique_hash_no_collisions(self):
        """Test unique hash generation with no existing hashes."""
        text = "unique document"
        existing_hashes = set()
        
        unique_hash = generate_unique_hash(text, existing_hashes)
        
        assert isinstance(unique_hash, int)
        assert unique_hash > 0
        assert unique_hash not in existing_hashes
    
    def test_generate_unique_hash_with_collisions(self):
        """Test unique hash generation with existing hash collisions."""
        text = "collision test"
        
        # Generate the first hash and add it to existing set
        first_hash = generate_text_hash(text, attempt=0)
        existing_hashes = {first_hash}
        
        # Generate unique hash should be different
        unique_hash = generate_unique_hash(text, existing_hashes)
        
        assert unique_hash != first_hash
        assert unique_hash not in existing_hashes
    
    def test_generate_unique_hash_max_attempts_error(self):
        """Test that max attempts raises an error."""
        text = "test"
        
        # Create a set with many hash values to force collision
        existing_hashes = set(range(1000000))  # Large set to force failures
        
        with pytest.raises(ValueError, match="Could not generate unique hash"):
            generate_unique_hash(text, existing_hashes)
    
    def test_hash_value_range(self):
        """Test that hash values are within expected range."""
        text = "range test"
        hash_value = generate_text_hash(text)
        
        # Should be positive and within 64-bit signed integer range
        assert 0 < hash_value < 2**63 - 1
    
    def test_empty_string_hash(self):
        """Test hashing empty string."""
        hash_value = generate_text_hash("")
        
        assert isinstance(hash_value, int)
        assert hash_value > 0
    
    def test_long_text_hash(self):
        """Test hashing very long text."""
        long_text = "A" * 10000
        hash_value = generate_text_hash(long_text)
        
        assert isinstance(hash_value, int)
        assert hash_value > 0