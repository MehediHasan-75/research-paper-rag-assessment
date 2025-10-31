import pytest
from unittest.mock import Mock, patch
from src.services.cache_service import CacheService

@pytest.fixture
def cache_service():
    """Create cache service with mocked Redis"""
    with patch('src.services.cache_service.redis.Redis'):
        service = CacheService()
        service.redis = Mock()
        return service

# ========== CACHING TESTS ==========

def test_cache_set_and_get(cache_service):
    """Test setting and getting cache"""
    result = {"answer": "test", "citations": []}
    
    cache_service.redis.get.return_value = None
    cache_service.set_query_cache("test query", result)
    
    assert cache_service.redis.setex.called

def test_cache_hit(cache_service):
    """Test cache hit"""
    import json
    result = {"answer": "test", "citations": []}
    
    cache_service.redis.get.return_value = json.dumps(result)
    cached = cache_service.get_query_cache("test query")
    
    assert cached is not None

def test_cache_miss(cache_service):
    """Test cache miss"""
    cache_service.redis.get.return_value = None
    cached = cache_service.get_query_cache("test query")
    
    assert cached is None

def test_cache_with_paper_ids(cache_service):
    """Test caching with specific paper IDs"""
    result = {"answer": "test"}
    paper_ids = [1, 2, 3]
    
    cache_service.set_query_cache("query", result, paper_ids=paper_ids)
    
    assert cache_service.redis.setex.called

def test_redis_unavailable(cache_service):
    """Test graceful fallback when Redis is unavailable"""
    cache_service.redis = None
    
    cache_service.set_query_cache("query", {})
    cached = cache_service.get_query_cache("query")
    
    assert cached is None