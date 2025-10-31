# test_embedding_service.py (with path fix)
import pytest
import sys
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create EmbeddingService instance"""
        with patch('src.services.embedding_service.SentenceTransformer'):
            service = EmbeddingService(
                model_name="test-model",
                dimension=384,
                batch_size=32
            )
            service.model = Mock()
            return service
    
    def test_initialization(self):
        """Test EmbeddingService initialization"""
        with patch('src.services.embedding_service.SentenceTransformer'):
            service = EmbeddingService(
                model_name="test-model",
                dimension=384,
                batch_size=32
            )
            assert service.model_name == "test-model"
            assert service.dimension == 384
            assert service.batch_size == 32
    
    def test_encode_single(self, embedding_service):
        """Test encoding single text"""
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        embedding_service.model.encode.return_value = mock_embedding
        
        result = embedding_service.encode_single("test text")
        
        assert isinstance(result, np.ndarray)
        embedding_service.model.encode.assert_called_once()
    
    def test_encode_batch(self, embedding_service):
        """Test encoding batch of texts"""
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        embedding_service.model.encode.return_value = mock_embeddings
        
        texts = ["text1", "text2", "text3"]
        result = embedding_service.encode_batch(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
    
    def test_encode_empty_batch(self, embedding_service):
        """Test encoding empty batch"""
        result = embedding_service.encode_batch([])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_similarity(self, embedding_service):
        """Test cosine similarity calculation"""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        
        similarity = embedding_service.similarity(emb1, emb2)
        
        assert similarity == 1.0
    
    def test_similarity_orthogonal(self, embedding_service):
        """Test similarity of orthogonal vectors"""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])
        
        similarity = embedding_service.similarity(emb1, emb2)
        
        assert abs(similarity) < 0.001
    
    def test_similarity_opposite_vectors(self, embedding_service):
        """Test similarity of opposite vectors"""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([-1.0, 0.0, 0.0])
        
        similarity = embedding_service.similarity(emb1, emb2)
        
        assert similarity == -1.0
    
    def test_get_dimension(self, embedding_service):
        """Test getting embedding dimension"""
        dim = embedding_service.get_dimension()
        assert dim == 384
    
    def test_get_model_name(self, embedding_service):
        """Test getting model name"""
        name = embedding_service.get_model_name()
        assert name == "test-model"
    
    def test_encode_with_custom_batch_size(self):
        """Test encoding with custom batch size"""
        with patch('src.services.embedding_service.SentenceTransformer'):
            service = EmbeddingService(
                model_name="test-model",
                dimension=384,
                batch_size=16
            )
            assert service.batch_size == 16
    
    def test_normalize_embeddings(self, embedding_service):
        """Test embedding normalization"""
        emb = np.array([3.0, 4.0])
        normalized = embedding_service.normalize_embedding(emb)
        
        # Check if norm is 1
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 0.001
    
    def test_distance_metric(self, embedding_service):
        """Test distance calculation between embeddings"""
        emb1 = np.array([0.0, 0.0])
        emb2 = np.array([3.0, 4.0])
        
        distance = embedding_service.euclidean_distance(emb1, emb2)
        
        assert distance == 5.0
