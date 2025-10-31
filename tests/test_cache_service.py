import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.cache_service import CacheService


class TestCacheService:
    """Test suite for CacheService"""
    
    @pytest.fixture
    def cache_service(self):
        """Create CacheService with mock Redis"""
        with patch('src.services.cache_service.redis.Redis'):
            service = CacheService(host="localhost", port=6379, ttl=3600)
            service.redis = Mock()
            return service
    
    def test_initialization_success(self):
        """Test successful cache initialization"""
        with patch('src.services.cache_service.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            
            service = CacheService()
            assert service.redis is not None
            assert service.ttl == 3600
    
    def test_initialization_failure(self):
        """Test cache initialization with Redis unavailable"""
        with patch('src.services.cache_service.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            service = CacheService()
            assert service.redis is None
    
    def test_get_query_cache_hit(self, cache_service):
        """Test cache hit"""
        cached_result = {"answer": "test", "confidence": 0.9}
        cache_service.redis.get.return_value = json.dumps(cached_result)
        
        result = cache_service.get_query_cache("test question")
        
        assert result is not None
        assert result['answer'] == "test"
    
    def test_get_query_cache_miss(self, cache_service):
        """Test cache miss"""
        cache_service.redis.get.return_value = None
        
        result = cache_service.get_query_cache("test question")
        
        assert result is None
    
    def test_set_query_cache(self, cache_service):
        """Test setting query cache"""
        result_data = {"answer": "test", "confidence": 0.9}
        
        cache_service.set_query_cache("test question", result_data)
        
        cache_service.redis.setex.assert_called_once()
        call_args = cache_service.redis.setex.call_args
        assert call_args[0][1] == 3600  # TTL
    
    def test_cache_with_paper_filter(self, cache_service):
        """Test cache with paper ID filter"""
        cache_service.redis.get.return_value = None
        
        paper_ids = [1, 2, 3]
        cache_service.get_query_cache("test", paper_ids=paper_ids)
        
        # Should generate different key for different filters
        key = cache_service._generate_key("test", paper_ids)
        assert isinstance(key, str)
        assert key.startswith("query:")
    
    def test_cache_disabled_when_redis_unavailable(self):
        """Test cache gracefully disabled when Redis unavailable"""
        with patch('src.services.cache_service.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Redis not available")
            service = CacheService()
            
            result = service.get_query_cache("test")
            assert result is None
    
    def test_clear_cache(self, cache_service):
        """Test clearing cache"""
        # FIXED: Mock keys() to return a proper list (not Mock)
        cache_service.redis.keys.return_value = [b'query:1', b'query:2', b'query:3']
        
        # Call clear
        cache_service.clear_query_cache()
        
        # Verify keys() was called with correct pattern
        cache_service.redis.keys.assert_called_once_with("query:*")
        
        # Verify delete was called with the keys
        cache_service.redis.delete.assert_called_once()
    
    def test_cache_expiration(self, cache_service):
        """Test cache TTL is set correctly"""
        service = CacheService(ttl=1800)
        assert service.ttl == 1800
    
    def test_cache_key_generation(self, cache_service):
        """Test cache key generation"""
        key1 = cache_service._generate_key("question1", None)
        key2 = cache_service._generate_key("question2", None)
        
        assert key1 != key2
        assert isinstance(key1, str)
        assert isinstance(key2, str)
    
    def test_invalid_json_in_cache(self, cache_service):
        """Test handling of invalid JSON in cache"""
        cache_service.redis.get.return_value = "invalid json {{"
        
        result = cache_service.get_query_cache("test")
        
        assert result is None
    
    def test_cache_with_complex_data(self, cache_service):
        """Test caching complex nested data structures"""
        complex_data = {
            "answer": "Complex answer",
            "citations": [
                {"paper_id": 1, "section": "Introduction"},
                {"paper_id": 2, "section": "Methods"}
            ],
            "metadata": {"confidence": 0.95, "sources": 2}
        }
        cache_service.redis.get.return_value = json.dumps(complex_data)
        
        result = cache_service.get_query_cache("complex query")
        
        assert result is not None
        assert len(result['citations']) == 2
        assert result['metadata']['confidence'] == 0.95
    
    def test_clear_cache_with_no_keys(self, cache_service):
        """Test clearing cache when no keys exist"""
        # Return empty list
        cache_service.redis.keys
