import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.services.embedding_service import EmbeddingService

class TestEmbeddingService:

    @pytest.fixture
    def embedding_service(self):
        with patch('sentence_transformers.SentenceTransformer'):
            service = EmbeddingService(
                model_name="test-model",
                dimension=384,
                batch_size=32,
            )
            service.model = Mock()
            return service

    def test_initialization(self):
        with patch('sentence_transformers.SentenceTransformer'):
            service = EmbeddingService(
                model_name="test-model",
                dimension=384,
                batch_size=32,
            )
            assert service.model_name == "test-model"
            assert service.dimension == 384
            assert service.batch_size == 32

    # ...rest of the tests as in the original...

    
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
    
