import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from src.services.qdrant_service import QdrantService

@pytest.fixture
def qdrant_service():
    """Mock Qdrant service for testing"""
    with patch('src.services.qdrant_service.QdrantClient'):
        service = QdrantService()
        service.client = MagicMock()
        return service

# ========== BASIC OPERATIONS ==========

def test_upsert_chunks_numpy(qdrant_service):
    """Test upserting numpy array embeddings"""
    chunks = [
        {'id': 1, 'text': 'Text 1', 'section': 'Intro', 'page_number': 1},
        {'id': 2, 'text': 'Text 2', 'section': 'Methods', 'page_number': 2},
    ]
    # ✅ Use numpy arrays
    embeddings = [
        np.random.randn(384).astype(np.float32),
        np.random.randn(384).astype(np.float32),
    ]
    
    vector_ids = qdrant_service.upsert_chunks(chunks, embeddings, paper_id=1)
    
    assert len(vector_ids) == 2

def test_upsert_chunks_lists(qdrant_service):
    """Test upserting list embeddings"""
    chunks = [
        {'id': 1, 'text': 'Text 1', 'section': 'Intro', 'page_number': 1},
    ]
    # ✅ Use lists
    embeddings = [
        [0.1] * 384,
    ]
    
    vector_ids = qdrant_service.upsert_chunks(chunks, embeddings, paper_id=1)
    
    assert len(vector_ids) == 1

def test_search_vectors(qdrant_service):
    """Test searching vectors"""
    query_vector = [0.1, 0.2] * 192
    
    mock_result = Mock()
    mock_result.payload = {
        'chunk_id': 1,
        'paper_id': 1,
        'text': 'Result text',
        'section': 'Methods',
        'page': 2
    }
    mock_result.score = 0.95
    
    qdrant_service.client.search.return_value = [mock_result]
    
    results = qdrant_service.search(query_vector, top_k=5)
    
    assert len(results) == 1
    assert results[0]['score'] == 0.95

def test_delete_by_paper(qdrant_service):
    """Test deleting vectors by paper ID"""
    qdrant_service.delete_by_paper(paper_id=1)
    
    assert qdrant_service.client.delete.called

# ========== EDGE CASES ==========

def test_empty_chunk_list(qdrant_service):
    """Test upserting empty chunk list"""
    chunks = []
    embeddings = []
    
    vector_ids = qdrant_service.upsert_chunks(chunks, embeddings, paper_id=1)
    
    assert vector_ids == []

def test_mixed_embeddings(qdrant_service):
    """Test handling mixed numpy and list embeddings"""
    chunks = [
        {'id': 1, 'text': 'Text 1', 'section': 'S', 'page_number': 1},
        {'id': 2, 'text': 'Text 2', 'section': 'S', 'page_number': 1},
    ]
    embeddings = [
        np.random.randn(384).astype(np.float32),  # Numpy
        [0.1] * 384,  # List
    ]
    
    vector_ids = qdrant_service.upsert_chunks(chunks, embeddings, paper_id=1)
    
    assert len(vector_ids) == 2
