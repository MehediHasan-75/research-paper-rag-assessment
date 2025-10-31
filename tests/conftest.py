# conftest.py (with path fix)
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_settings():
    """Configure settings for tests"""
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['QDRANT_HOST'] = 'localhost'
    os.environ['REDIS_HOST'] = 'localhost'


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module imports between tests"""
    yield


@pytest.fixture
def sample_paper_dict():
    """Sample paper data for testing"""
    return {
        'paper_name': 'test_paper_2024',
        'title': 'Test Research Paper',
        'authors': ['Author One', 'Author Two'],
        'year': 2024,
        'filename': 'test_paper_2024.pdf',
        'file_path': '/uploads/test.pdf',
        'total_pages': 15,
        'abstract': 'This is a test abstract',
        'quality_score': 0.85,
        'keywords': ['test', 'research'],
        'format_type': 'standard'
    }


@pytest.fixture
def sample_chunk_dict():
    """Sample chunk data for testing"""
    return {
        'paper_id': 1,
        'chunk_index': 0,
        'text': 'This is sample chunk text for testing purposes',
        'section': 'Introduction',
        'page_number': 1,
        'section_id': '1.0',
        'section_level': 0
    }


@pytest.fixture
def sample_query_dict():
    """Sample query data for testing"""
    return {
        'query_text': 'What is LWE cryptography?',
        'answer': 'LWE stands for Learning with Error',
        'top_k': 5,
        'response_time': 0.45,
        'confidence': 0.92,
        'user_rating': 4
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    import numpy as np
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def sample_search_result():
    """Sample search result from Qdrant"""
    return {
        'paper_id': 1,
        'chunk_id': 5,
        'text': 'LWE is a mathematical problem used in cryptography',
        'section': 'Introduction',
        'page': 2,
        'score': 0.92,
        'metadata': {'paper_name': 'test_paper'}
    }


@pytest.fixture
def sample_citation():
    """Sample citation data"""
    return {
        'query_id': 1,
        'paper_id': 2,
        'chunk_id': 10,
        'relevance_score': 0.89,
        'text_snippet': 'Relevant text from paper'
    }


@pytest.fixture
def mock_pdf_path(tmp_path):
    """Create temporary PDF file for testing"""
    pdf_file = tmp_path / "test_paper.pdf"
    pdf_file.write_text("Mock PDF content")
    return str(pdf_file)


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama LLM"""
    return {
        'message': {
            'content': 'This is a generated answer based on the retrieved context.'
        },
        'eval_count': 100,
        'eval_duration': 500000000
    }


@pytest.fixture
def mock_sentence_transformer_model():
    """Mock SentenceTransformer model"""
    mock_model = Mock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture(scope="function")
def cleanup_cache():
    """Cleanup cache after each test"""
    yield
    # Cleanup code here
    pass


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking"""
    import time
    
    class BenchmarkTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return BenchmarkTimer()


# Markers for organizing tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database"
    )


# Pytest hooks for enhanced reporting
def pytest_runtest_logreport(report):
    """Hook for test report logging"""
    if report.when == "call":
        pass  # Custom logging here


# Test collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Auto-mark tests based on their module
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "database" in str(item.fspath):
            item.add_marker(pytest.mark.database)
