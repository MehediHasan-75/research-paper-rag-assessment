# test_rag_pipeline.py (with path fix)
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """Test suite for RAG Pipeline"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for RAG pipeline"""
        qdrant_service = Mock()
        embedding_service = Mock()
        db_session = Mock()
        return qdrant_service, embedding_service, db_session
    
    @pytest.fixture
    def rag_pipeline(self, mock_dependencies):
        """Create RAG pipeline instance with mocks"""
        qdrant_service, embedding_service, db_session = mock_dependencies
        return RAGPipeline(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            db_session=db_session,
            model="test-model"
        )
    
    def test_initialization(self, mock_dependencies):
        """Test RAG pipeline initialization"""
        qdrant_service, embedding_service, db_session = mock_dependencies
        pipeline = RAGPipeline(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            db_session=db_session
        )
        assert pipeline.model == "deepseek-r1:8b"
        assert pipeline.qdrant is not None
        assert pipeline.embedder is not None
        assert pipeline.db is not None
    
    def test_generate_answer_no_results(self, rag_pipeline):
        """Test generate_answer when no results found"""
        rag_pipeline.embedder.encode_single.return_value = [0.1, 0.2, 0.3]
        rag_pipeline.qdrant.search.return_value = []
        
        result = rag_pipeline.generate_answer("test question")
        
        assert result['answer'] == "No relevant information found in the papers."
        assert result['citations'] == []
        assert result['sources_used'] == []
        assert result['confidence'] == 0.0
        assert result['response_time'] > 0
    
    def test_generate_answer_with_results(self, rag_pipeline):
        """Test generate_answer with valid results"""
        mock_paper = Mock()
        mock_paper.id = 1
        mock_paper.title = "Test Paper"
        mock_paper.paper_name = "test_paper"
        
        rag_pipeline.embedder.encode_single.return_value = [0.1, 0.2, 0.3]
        rag_pipeline.qdrant.search.return_value = [
            {
                'paper_id': 1,
                'text': 'Sample context',
                'section': 'Introduction',
                'page': 1,
                'score': 0.95
            }
        ]
        rag_pipeline.db.query.return_value.filter.return_value.first.return_value = mock_paper
        
        with patch('ollama.chat') as mock_ollama:
            mock_ollama.return_value = {
                'message': {'content': 'This is a test answer'}
            }
            result = rag_pipeline.generate_answer("test question", top_k=5)
        
        assert 'answer' in result
        assert len(result['citations']) > 0
        assert result['confidence'] > 0
        assert result['response_time'] > 0
    
    def test_generate_answer_paper_filtering(self, rag_pipeline):
        """Test generate_answer with paper ID filtering"""
        rag_pipeline.embedder.encode_single.return_value = [0.1, 0.2]
        rag_pipeline.qdrant.search.return_value = []
        
        paper_ids = [1, 2, 3]
        rag_pipeline.generate_answer("test", paper_ids=paper_ids)
        
        rag_pipeline.qdrant.search.assert_called_once()
        call_kwargs = rag_pipeline.qdrant.search.call_args[1]
        assert call_kwargs['paper_ids'] == paper_ids
    
    def test_generate_answer_top_k_parameter(self, rag_pipeline):
        """Test top_k parameter is passed correctly"""
        rag_pipeline.embedder.encode_single.return_value = [0.1]
        rag_pipeline.qdrant.search.return_value = []
        
        rag_pipeline.generate_answer("test", top_k=10)
        
        call_kwargs = rag_pipeline.qdrant.search.call_args[1]
        assert call_kwargs['top_k'] == 10
    
    def test_confidence_calculation(self, rag_pipeline):
        """Test confidence is calculated as average of top 3 scores"""
        mock_paper = Mock()
        mock_paper.title = "Test"
        mock_paper.paper_name = "test"
        
        rag_pipeline.embedder.encode_single.return_value = [0.1]
        rag_pipeline.qdrant.search.return_value = [
            {'paper_id': 1, 'text': 'a', 'section': 's', 'page': 1, 'score': 0.9},
            {'paper_id': 1, 'text': 'b', 'section': 's', 'page': 1, 'score': 0.8},
            {'paper_id': 1, 'text': 'c', 'section': 's', 'page': 1, 'score': 0.7},
            {'paper_id': 1, 'text': 'd', 'section': 's', 'page': 1, 'score': 0.5},
        ]
        rag_pipeline.db.query.return_value.filter.return_value.first.return_value = mock_paper
        
        with patch('ollama.chat'):
            result = rag_pipeline.generate_answer("test")
        
        expected_confidence = (0.9 + 0.8 + 0.7) / 3
        assert result['confidence'] == round(expected_confidence, 2)
