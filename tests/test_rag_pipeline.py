# tests/test_rag_pipeline.py
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import numpy as np
from typing import List, Dict

# Import the RAG pipeline - adjust path as needed
import sys
sys.path.insert(0, 'src')
from services.rag_pipeline import RAGPipeline


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant service"""
    mock = Mock()
    mock.search = Mock()
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    mock = Mock()
    mock.encode_single = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    return mock


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    mock = Mock()
    return mock


@pytest.fixture
def mock_paper():
    """Mock paper object"""
    paper = Mock()
    paper.id = 1
    paper.title = "Advanced Cryptography Techniques"
    return paper


@pytest.fixture
def rag_pipeline(mock_qdrant, mock_embedding_service, mock_db_session):
    """Create RAG pipeline instance with mocked dependencies"""
    return RAGPipeline(
        qdrant_service=mock_qdrant,
        embedding_service=mock_embedding_service,
        db_session=mock_db_session,
        model="deepseek-r1:8b"
    )


@pytest.fixture
def sample_search_results():
    """Sample search results from Qdrant"""
    return [
        {
            'paper_id': 1,
            'text': 'Learning with Errors is a fundamental problem in lattice-based cryptography.',
            'section': 'Introduction',
            'page': 1,
            'score': 0.95
        },
        {
            'paper_id': 2,
            'text': 'The LWE problem provides security for homomorphic encryption schemes.',
            'section': 'Background',
            'page': 3,
            'score': 0.87
        },
        {
            'paper_id': 1,
            'text': 'Parameter selection affects the computational complexity significantly.',
            'section': 'Analysis',
            'page': 8,
            'score': 0.82
        },
        {
            'paper_id': 3,
            'text': 'Data compression reduces the overhead in practical implementations.',
            'section': 'Optimization',
            'page': 12,
            'score': 0.79
        },
        {
            'paper_id': 2,
            'text': 'Strassen multiplication improves matrix operation efficiency.',
            'section': 'Techniques',
            'page': 15,
            'score': 0.75
        }
    ]


# ============================================================================
# TEST CLASS: Initialization
# ============================================================================

class TestRAGPipelineInitialization:
    """Test RAG pipeline initialization"""

    def test_pipeline_initialization(self, rag_pipeline, mock_qdrant, 
                                     mock_embedding_service, mock_db_session):
        """Test that pipeline initializes with correct dependencies"""
        assert rag_pipeline.qdrant is mock_qdrant
        assert rag_pipeline.embedder is mock_embedding_service
        assert rag_pipeline.db is mock_db_session
        assert rag_pipeline.model == "deepseek-r1:8b"

    def test_custom_model_initialization(self, mock_qdrant, mock_embedding_service, 
                                        mock_db_session):
        """Test pipeline initialization with custom model"""
        custom_model = "custom-model:7b"
        pipeline = RAGPipeline(
            qdrant_service=mock_qdrant,
            embedding_service=mock_embedding_service,
            db_session=mock_db_session,
            model=custom_model
        )
        assert pipeline.model == custom_model


# ============================================================================
# TEST CLASS: Query Encoding
# ============================================================================

class TestQueryEncoding:
    """Test query encoding functionality"""

    def test_query_encoding_called(self, rag_pipeline, mock_embedding_service, 
                                   mock_qdrant, sample_search_results):
        """Test that embedder is called to encode the question"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            question = "What is Learning with Errors?"
            rag_pipeline.generate_answer(question)
            
            mock_embedding_service.encode_single.assert_called_once_with(question)

    def test_query_encoding_vector_format(self, rag_pipeline, mock_embedding_service, 
                                          mock_qdrant, sample_search_results):
        """Test that encoded query vector is converted to list format"""
        test_vector = np.array([0.1, 0.2, 0.3])
        mock_embedding_service.encode_single.return_value = test_vector
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            rag_pipeline.generate_answer("Test question")
            
            # Verify search was called with vector as list
            mock_qdrant.search.assert_called_once()
            call_args = mock_qdrant.search.call_args
            assert isinstance(call_args.kwargs['query_vector'], list)
            np.testing.assert_array_equal(call_args.kwargs['query_vector'], test_vector.tolist())


# ============================================================================
# TEST CLASS: Vector Search and Retrieval
# ============================================================================

class TestVectorSearchRetrieval:
    """Test vector search and chunk retrieval"""

    def test_qdrant_search_called_with_correct_params(self, rag_pipeline, 
                                                      mock_embedding_service, 
                                                      mock_qdrant, sample_search_results):
        """Test that Qdrant search is called with correct parameters"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            question = "What is LWE?"
            rag_pipeline.generate_answer(question, top_k=5)
            
            mock_qdrant.search.assert_called_once()
            call_kwargs = mock_qdrant.search.call_args.kwargs
            assert call_kwargs['top_k'] == 5
            assert call_kwargs['paper_ids'] is None

    def test_qdrant_search_with_paper_filter(self, rag_pipeline, mock_embedding_service, 
                                             mock_qdrant, sample_search_results):
        """Test that paper_ids filter is passed to Qdrant"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            paper_ids = [1, 2, 3]
            rag_pipeline.generate_answer("Test question", paper_ids=paper_ids)
            
            call_kwargs = mock_qdrant.search.call_args.kwargs
            assert call_kwargs['paper_ids'] == paper_ids

    def test_empty_search_results_handling(self, rag_pipeline, mock_embedding_service, 
                                          mock_qdrant):
        """Test handling of empty search results"""
        mock_qdrant.search.return_value = []
        
        result = rag_pipeline.generate_answer("Irrelevant question")
        
        assert result['answer'] == "No relevant information found in the papers."
        assert result['citations'] == []
        assert result['sources_used'] == []
        assert result['confidence'] == 0.0
        assert 'response_time' in result

    def test_custom_top_k(self, rag_pipeline, mock_embedding_service, 
                         mock_qdrant, sample_search_results):
        """Test custom top_k parameter"""
        mock_qdrant.search.return_value = sample_search_results[:3]
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            rag_pipeline.generate_answer("Question", top_k=3)
            
            assert mock_qdrant.search.call_args.kwargs['top_k'] == 3


# ============================================================================
# TEST CLASS: Context Building
# ============================================================================

class TestContextBuilding:
    """Test context construction from search results"""

    def test_context_includes_all_results(self, rag_pipeline, mock_embedding_service, 
                                         mock_qdrant, mock_db_session, sample_search_results):
        """Test that context includes all retrieved results"""
        mock_qdrant.search.return_value = sample_search_results
        
        # Setup mock paper queries
        mock_papers = {
            1: Mock(title="Paper 1", id=1),
            2: Mock(title="Paper 2", id=2),
            3: Mock(title="Paper 3", id=3)
        }
        
        def mock_query_side_effect(model):
            mock_query = Mock()
            mock_query.filter = Mock(return_value=Mock(first=Mock(
                side_effect=lambda: mock_papers.get(model.id)
            )))
            return mock_query
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                # Capture prompt to verify context
                rag_pipeline.generate_answer("Test question")
                
                # Verify ollama.chat was called
                assert mock_chat.called
                prompt = mock_chat.call_args[1]['messages'][0]['content']
                
                # Verify context elements are in prompt
                assert 'Source 1' in prompt
                assert 'Source 2' in prompt
                assert 'Source 3' in prompt
                assert 'Learning with Errors' in prompt

    def test_context_format_correctness(self, rag_pipeline, mock_embedding_service, 
                                       mock_qdrant, sample_search_results):
        """Test that context is formatted correctly"""
        mock_qdrant.search.return_value = sample_search_results[:2]
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Test answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                mock_paper_obj = Mock(title="Test Paper")
                MockPaper.return_value = mock_paper_obj
                
                rag_pipeline.generate_answer("Question")
                
                prompt = mock_chat.call_args[1]['messages'][0]['content']
                
                # Verify format structure
                assert '[Source' in prompt
                assert 'Page' in prompt
                assert 'From' in prompt


# ============================================================================
# TEST CLASS: LLM Answer Generation
# ============================================================================

class TestLLMAnswerGeneration:
    """Test answer generation with LLM"""

    @patch('services.rag_pipeline.ollama.chat')
    def test_ollama_chat_called_correctly(self, mock_chat, rag_pipeline, 
                                         mock_embedding_service, mock_qdrant, 
                                         sample_search_results):
        """Test that Ollama is called with correct model and format"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Generated answer'}}
        
        rag_pipeline.generate_answer("Test question")
        
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args[1]['model'] == 'deepseek-r1:8b'
        assert isinstance(call_args[1]['messages'], list)
        assert call_args[1]['messages'][0]['role'] == 'user'

    @patch('services.rag_pipeline.ollama.chat')
    def test_answer_extraction(self, mock_chat, rag_pipeline, mock_embedding_service, 
                              mock_qdrant, sample_search_results):
        """Test that answer is correctly extracted from response"""
        mock_qdrant.search.return_value = sample_search_results
        expected_answer = "LWE is a fundamental problem in lattice-based cryptography."
        mock_chat.return_value = {'message': {'content': expected_answer}}
        
        result = rag_pipeline.generate_answer("What is LWE?")
        
        assert result['answer'] == expected_answer

    @patch('services.rag_pipeline.ollama.chat')
    def test_prompt_includes_context_and_question(self, mock_chat, rag_pipeline, 
                                                  mock_embedding_service, mock_qdrant, 
                                                  sample_search_results):
        """Test that prompt includes both context and original question"""
        mock_qdrant.search.return_value = sample_search_results[:1]
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        test_question = "What are the applications of LWE?"
        rag_pipeline.generate_answer(test_question)
        
        prompt = mock_chat.call_args[1]['messages'][0]['content']
        assert test_question in prompt
        assert 'Context from research papers' in prompt


# ============================================================================
# TEST CLASS: Citations Processing
# ============================================================================

class TestCitationsProcessing:
    """Test citation extraction and formatting"""

    def test_citations_format(self, rag_pipeline, mock_embedding_service, 
                             mock_qdrant, sample_search_results):
        """Test that citations have correct format"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                # Create mock papers
                papers = {}
                for result in sample_search_results:
                    if result['paper_id'] not in papers:
                        mock_paper = Mock()
                        mock_paper.id = result['paper_id']
                        mock_paper.title = f"Paper {result['paper_id']}"
                        papers[result['paper_id']] = mock_paper
                
                result = rag_pipeline.generate_answer("Question")
                
                assert isinstance(result['citations'], list)
                assert len(result['citations']) > 0
                
                for citation in result['citations']:
                    assert 'paper_title' in citation
                    assert 'section' in citation
                    assert 'page' in citation
                    assert 'relevance_score' in citation

    def test_sources_used_uniqueness(self, rag_pipeline, mock_embedding_service, 
                                    mock_qdrant, sample_search_results):
        """Test that sources_used contains unique paper titles"""
        # sample_search_results has duplicate paper IDs
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                result = rag_pipeline.generate_answer("Question")
                
                # Should have unique sources
                assert len(result['sources_used']) <= len(sample_search_results)
                # Check for duplicates
                assert len(result['sources_used']) == len(set(result['sources_used']))

    def test_relevance_score_rounding(self, rag_pipeline, mock_embedding_service, 
                                     mock_qdrant, sample_search_results):
        """Test that relevance scores are rounded to 3 decimals"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                result = rag_pipeline.generate_answer("Question")
                
                for citation in result['citations']:
                    score_str = str(citation['relevance_score'])
                    # Check that score has max 3 decimal places
                    if '.' in score_str:
                        decimals = len(score_str.split('.')[1])
                        assert decimals <= 3


# ============================================================================
# TEST CLASS: Confidence Calculation
# ============================================================================

class TestConfidenceCalculation:
    """Test confidence score calculation"""

    def test_confidence_from_top_3_scores(self, rag_pipeline, mock_embedding_service, 
                                         mock_qdrant):
        """Test that confidence is calculated from top 3 scores"""
        search_results = [
            {'paper_id': 1, 'text': 'Text 1', 'section': 'S1', 'page': 1, 'score': 0.9},
            {'paper_id': 2, 'text': 'Text 2', 'section': 'S2', 'page': 2, 'score': 0.8},
            {'paper_id': 3, 'text': 'Text 3', 'section': 'S3', 'page': 3, 'score': 0.7},
            {'paper_id': 4, 'text': 'Text 4', 'section': 'S4', 'page': 4, 'score': 0.5},
        ]
        mock_qdrant.search.return_value = search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                result = rag_pipeline.generate_answer("Question")
                
                # Expected: (0.9 + 0.8 + 0.7) / 3 = 0.8
                expected_confidence = round((0.9 + 0.8 + 0.7) / 3, 2)
                assert result['confidence'] == expected_confidence

    def test_confidence_with_fewer_than_3_results(self, rag_pipeline, mock_embedding_service, 
                                                 mock_qdrant):
        """Test confidence calculation when fewer than 3 results"""
        search_results = [
            {'paper_id': 1, 'text': 'Text 1', 'section': 'S1', 'page': 1, 'score': 0.9},
            {'paper_id': 2, 'text': 'Text 2', 'section': 'S2', 'page': 2, 'score': 0.7},
        ]
        mock_qdrant.search.return_value = search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            with patch('services.rag_pipeline.Paper') as MockPaper:
                result = rag_pipeline.generate_answer("Question")
                
                # Expected: (0.9 + 0.7) / 2 = 0.8
                expected_confidence = round((0.9 + 0.7) / 2, 2)
                assert result['confidence'] == expected_confidence

    def test_confidence_zero_for_empty_results(self, rag_pipeline, mock_qdrant):
        """Test confidence is 0 when no results"""
        mock_qdrant.search.return_value = []
        
        result = rag_pipeline.generate_answer("Question")
        
        assert result['confidence'] == 0.0


# ============================================================================
# TEST CLASS: Response Timing
# ============================================================================

class TestResponseTiming:
    """Test response time tracking"""

    @patch('services.rag_pipeline.time.time')
    @patch('services.rag_pipeline.ollama.chat')
    def test_response_time_tracking(self, mock_chat, mock_time, rag_pipeline, 
                                   mock_embedding_service, mock_qdrant, 
                                   sample_search_results):
        """Test that response time is tracked and included in output"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        # Mock time progression
        mock_time.side_effect = [100.0, 102.5]  # 2.5 seconds elapsed
        
        result = rag_pipeline.generate_answer("Question")
        
        assert 'response_time' in result
        assert result['response_time'] == 2.5

    @patch('services.rag_pipeline.ollama.chat')
    def test_response_time_is_positive(self, mock_chat, rag_pipeline, 
                                      mock_embedding_service, mock_qdrant, 
                                      sample_search_results):
        """Test that response time is positive"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        result = rag_pipeline.generate_answer("Question")
        
        assert result['response_time'] >= 0

    @patch('services.rag_pipeline.ollama.chat')
    def test_response_time_rounded(self, mock_chat, rag_pipeline, 
                                  mock_embedding_service, mock_qdrant, 
                                  sample_search_results):
        """Test that response time is rounded to 2 decimals"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        result = rag_pipeline.generate_answer("Question")
        
        time_str = str(result['response_time'])
        if '.' in time_str:
            decimals = len(time_str.split('.')[1])
            assert decimals <= 2


# ============================================================================
# TEST CLASS: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline"""

    @patch('services.rag_pipeline.ollama.chat')
    def test_full_pipeline_flow(self, mock_chat, rag_pipeline, mock_embedding_service, 
                               mock_qdrant, sample_search_results):
        """Test complete pipeline flow from question to answer"""
        mock_embedding_service.encode_single.return_value = np.array([0.1, 0.2])
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Comprehensive answer'}}
        
        result = rag_pipeline.generate_answer("What is LWE cryptography?", top_k=5)
        
        # Verify all expected fields are present
        assert 'answer' in result
        assert 'citations' in result
        assert 'sources_used' in result
        assert 'confidence' in result
        assert 'response_time' in result
        
        # Verify types
        assert isinstance(result['answer'], str)
        assert isinstance(result['citations'], list)
        assert isinstance(result['sources_used'], list)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['response_time'], float)

    @patch('services.rag_pipeline.ollama.chat')
    def test_pipeline_with_all_parameters(self, mock_chat, rag_pipeline, 
                                         mock_embedding_service, mock_qdrant, 
                                         sample_search_results):
        """Test pipeline with all optional parameters"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        result = rag_pipeline.generate_answer(
            question="LWE applications?",
            top_k=10,
            paper_ids=[1, 2, 3]
        )
        
        assert result is not None
        assert 'answer' in result
        
        # Verify parameters were passed correctly
        call_kwargs = mock_qdrant.search.call_args.kwargs
        assert call_kwargs['top_k'] == 10
        assert call_kwargs['paper_ids'] == [1, 2, 3]

    @patch('services.rag_pipeline.ollama.chat')
    def test_error_handling_on_db_query_failure(self, mock_chat, rag_pipeline, 
                                               mock_embedding_service, mock_qdrant, 
                                               sample_search_results):
        """Test behavior when database query fails"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        # Make db.query raise an exception
        rag_pipeline.db.query = Mock(side_effect=Exception("DB Connection Error"))
        
        with pytest.raises(Exception):
            rag_pipeline.generate_answer("Question")


# ============================================================================
# TEST CLASS: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_question_string(self, rag_pipeline, mock_embedding_service, 
                                  mock_qdrant):
        """Test handling of empty question"""
        mock_embedding_service.encode_single.return_value = np.array([0.0])
        mock_qdrant.search.return_value = []
        
        result = rag_pipeline.generate_answer("")
        
        assert result['answer'] == "No relevant information found in the papers."

    @patch('services.rag_pipeline.ollama.chat')
    def test_very_long_question(self, mock_chat, rag_pipeline, mock_embedding_service, 
                               mock_qdrant, sample_search_results):
        """Test handling of very long question"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        long_question = "What is the relationship between " * 50
        result = rag_pipeline.generate_answer(long_question)
        
        assert 'answer' in result
        mock_qdrant.search.assert_called_once()

    @patch('services.rag_pipeline.ollama.chat')
    def test_single_search_result(self, mock_chat, rag_pipeline, mock_embedding_service, 
                                 mock_qdrant):
        """Test handling of single search result"""
        search_result = [{
            'paper_id': 1,
            'text': 'Single result',
            'section': 'Introduction',
            'page': 1,
            'score': 0.95
        }]
        mock_qdrant.search.return_value = search_result
        mock_chat.return_value = {'message': {'content': 'Answer based on single source'}}
        
        result = rag_pipeline.generate_answer("Question")
        
        assert len(result['citations']) == 1
        assert result['confidence'] == round(0.95, 2)

    @patch('services.rag_pipeline.ollama.chat')
    def test_special_characters_in_question(self, mock_chat, rag_pipeline, 
                                           mock_embedding_service, mock_qdrant, 
                                           sample_search_results):
        """Test handling of special characters in question"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        question = "What is LWE? @#$%^&*() [test] {question}"
        result = rag_pipeline.generate_answer(question)
        
        assert 'answer' in result
        mock_qdrant.search.assert_called_once()

    @patch('services.rag_pipeline.ollama.chat')
    def test_unicode_in_question(self, mock_chat, rag_pipeline, mock_embedding_service, 
                                mock_qdrant, sample_search_results):
        """Test handling of unicode characters"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        question = "What is Learning with Errors? 中文 العربية"
        result = rag_pipeline.generate_answer(question)
        
        assert 'answer' in result

    def test_top_k_zero(self, rag_pipeline, mock_embedding_service, mock_qdrant):
        """Test with top_k = 0"""
        mock_qdrant.search.return_value = []
        
        result = rag_pipeline.generate_answer("Question", top_k=0)
        
        assert mock_qdrant.search.call_args.kwargs['top_k'] == 0

    def test_negative_paper_ids(self, rag_pipeline, mock_embedding_service, 
                               mock_qdrant, sample_search_results):
        """Test with negative paper IDs"""
        mock_qdrant.search.return_value = sample_search_results
        
        with patch('services.rag_pipeline.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Answer'}}
            
            paper_ids = [-1, -2, -3]
            result = rag_pipeline.generate_answer("Question", paper_ids=paper_ids)
            
            assert mock_qdrant.search.call_args.kwargs['paper_ids'] == paper_ids


# ============================================================================
# TEST CLASS: Data Integrity
# ============================================================================

class TestDataIntegrity:
    """Test data integrity and consistency"""

    @patch('services.rag_pipeline.ollama.chat')
    def test_citations_match_search_results(self, mock_chat, rag_pipeline, 
                                           mock_embedding_service, mock_qdrant, 
                                           sample_search_results):
        """Test that citations match the search results"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        with patch('services.rag_pipeline.Paper') as MockPaper:
            result = rag_pipeline.generate_answer("Question")
            
            # Number of citations should match number of results
            assert len(result['citations']) == len(sample_search_results)

    @patch('services.rag_pipeline.ollama.chat')
    def test_sources_used_from_citations(self, mock_chat, rag_pipeline, 
                                        mock_embedding_service, mock_qdrant, 
                                        sample_search_results):
        """Test that sources_used are derived from citations"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Answer'}}
        
        with patch('services.rag_pipeline.Paper') as MockPaper:
            result = rag_pipeline.generate_answer("Question")
            
            citation_titles = set(c['paper_title'] for c in result['citations'])
            sources = set(result['sources_used'])
            
            # Sources should be subset of or equal to citation titles
            assert sources == citation_titles or len(sources) <= len(citation_titles)

    @patch('services.rag_pipeline.ollama.chat')
    def test_answer_not_empty(self, mock_chat, rag_pipeline, mock_embedding_service, 
                             mock_qdrant, sample_search_results):
        """Test that answer is not empty for valid search results"""
        mock_qdrant.search.return_value = sample_search_results
        mock_chat.return_value = {'message': {'content': 'Valid answer from LLM'}}
        
        result = rag_pipeline.generate_answer("Question")
        
        assert result['answer'] != ""
        assert result['answer'] != None


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-ra'])
