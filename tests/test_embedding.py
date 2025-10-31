import pytest
import numpy as np
from src.services.embedding_service import EmbeddingService

@pytest.fixture
def embedding_service():
    """Create embedding service"""
    return EmbeddingService()

# ========== TESTS ==========

def test_embedding_service_initialization(embedding_service):
    """Test init"""
    assert embedding_service is not None
    assert embedding_service.dimension == 384

def test_encode_single_text(embedding_service):
    """Test single encoding"""
    text = "This is a test sentence."
    embedding = embedding_service.encode_single(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)

def test_encode_batch(embedding_service):
    """Test batch"""
    texts = ["First.", "Second.", "Third."]
    embeddings = embedding_service.encode_batch(texts)
    
    assert embeddings.shape == (3, 384)

def test_encode_consistency(embedding_service):
    """Test consistency"""
    text = "Consistent test."
    emb1 = embedding_service.encode_single(text)
    emb2 = embedding_service.encode_single(text)
    
    assert np.allclose(emb1, emb2)

def test_encode_different_texts(embedding_service):
    """Test different"""
    text1 = "Machine learning"
    text2 = "Cooking recipes"
    
    emb1 = embedding_service.encode_single(text1)
    emb2 = embedding_service.encode_single(text2)
    
    assert not np.allclose(emb1, emb2)

def test_encode_empty_string(embedding_service):
    """Test empty"""
    embedding = embedding_service.encode_single("")
    assert embedding.shape == (384,)

def test_encode_long_text(embedding_service):
    """Test long"""
    text = " ".join(["word"] * 1000)
    embedding = embedding_service.encode_single(text)
    assert embedding.shape == (384,)

def test_encode_special_characters(embedding_service):
    """Test special"""
    text = "Special: @#$%^&*()_+-=[]{}|;:',.<>?/`~"
    embedding = embedding_service.encode_single(text)
    assert embedding.shape == (384,)

def test_encode_unicode(embedding_service):
    """Test unicode"""
    text = "中文 عربي текст 😀"
    embedding = embedding_service.encode_single(text)
    assert embedding.shape == (384,)

def test_batch_large(embedding_service):
    """Test large batch"""
    texts = [f"Text {i}" for i in range(100)]
    embeddings = embedding_service.encode_batch(texts)
    assert embeddings.shape == (100, 384)

def test_batch_single(embedding_service):
    """Test single batch"""
    embeddings = embedding_service.encode_batch(["Single"])
    assert embeddings.shape == (1, 384)

def test_batch_empty(embedding_service):
    """Test empty batch"""
    embeddings = embedding_service.encode_batch([])
    assert embeddings.shape[0] == 0
