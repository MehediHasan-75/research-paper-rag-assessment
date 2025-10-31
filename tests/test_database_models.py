# test_database_models.py (with path fix)
import pytest
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import Paper, Chunk, QueryHistory, Citation


class TestPaperModel:
    """Test suite for Paper model"""
    
    def test_paper_creation(self):
        """Test creating a Paper instance"""
        paper = Paper(
            paper_name="test_paper",
            title="Test Paper Title",
            authors=["Author 1", "Author 2"],
            year=2024,
            filename="test.pdf",
            file_path="/path/to/file.pdf",
            total_pages=20,
            quality_score=0.85
        )
        assert paper.paper_name == "test_paper"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.quality_score == 0.85
    
    def test_quality_score_validation_valid(self):
        """Test quality_score validation with valid values"""
        paper = Paper(
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf"
        )
        assert paper.quality_score == 0.0  # Default value
        
        paper.quality_score = 0.5
        assert paper.quality_score == 0.5
        
        paper.quality_score = 1.0
        assert paper.quality_score == 1.0
    
    def test_quality_score_validation_invalid(self):
        """Test quality_score validation rejects invalid values"""
        paper = Paper(
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf"
        )
        with pytest.raises(ValueError):
            paper.quality_score = 1.5
        
        with pytest.raises(ValueError):
            paper.quality_score = -0.1
    
    def test_quality_score_none_handling(self):
        """Test quality_score None is converted to 0.0"""
        paper = Paper(
            paper_name="test1",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf"
        )
        paper.quality_score = None
        assert paper.quality_score == 0.0
    
    def test_year_validation_valid(self):
        """Test year validation with valid years"""
        paper = Paper(
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf",
            year=2024
        )
        assert paper.year == 2024
    
    def test_year_validation_invalid(self):
        """Test year validation rejects invalid years"""
        paper = Paper(
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf"
        )
        # Year > 2150 should return None
        paper.year = 2200
        assert paper.year is None
        
        # Negative year should return None
        paper.year = -100
        assert paper.year is None
    
    def test_year_string_conversion(self):
        """Test year accepts string and converts to int"""
        paper = Paper(
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf"
        )
        paper.year = "2024"
        assert paper.year == 2024
    
    def test_paper_repr(self):
        """Test Paper string representation"""
        paper = Paper(
            id=1,
            paper_name="test",
            title="Test",
            filename="test.pdf",
            file_path="/test.pdf",
            quality_score=0.75
        )
        repr_str = repr(paper)
        assert "test" in repr_str
        assert "0.75" in repr_str


class TestChunkModel:
    """Test suite for Chunk model"""
    
    def test_chunk_creation(self):
        """Test creating a Chunk instance"""
        chunk = Chunk(
            paper_id=1,
            chunk_index=0,
            text="Sample chunk text",
            section="Introduction",
            page_number=1,
            section_id="1.1",
            section_level=1
        )
        assert chunk.paper_id == 1
        assert chunk.text == "Sample chunk text"
        assert chunk.section_id == "1.1"
        assert chunk.section_level == 1
    
    def test_chunk_repr(self):
        """Test Chunk string representation"""
        chunk = Chunk(
            id=1,
            paper_id=1,
            chunk_index=0,
            text="text",
            section="s",
            page_number=1,
            section_id="1.1"
        )
        repr_str = repr(chunk)
        assert "1.1" in repr_str


class TestQueryHistoryModel:
    """Test suite for QueryHistory model"""
    
    def test_query_history_creation(self):
        """Test creating QueryHistory instance"""
        query = QueryHistory(
            query_text="What is LWE?",
            answer="LWE is Learning with Error",
            top_k=5,
            response_time=0.5,
            confidence=0.95,
            user_rating=4
        )
        assert query.query_text == "What is LWE?"
        assert query.confidence == 0.95
        assert query.user_rating == 4
    
    def test_query_history_paper_filter(self):
        """Test QueryHistory with paper_filter"""
        query = QueryHistory(
            query_text="test",
            top_k=5,
            response_time=0.5,
            confidence=0.8,
            paper_filter=[1, 2, 3]
        )
        assert query.paper_filter == [1, 2, 3]
    
    def test_query_history_repr(self):
        """Test QueryHistory string representation"""
        query = QueryHistory(
            id=1,
            query_text="What is cryptography?",
            response_time=0.5,
            confidence=0.85
        )
        repr_str = repr(query)
        assert "cryptography" in repr_str


class TestCitationModel:
    """Test suite for Citation model"""
    
    def test_citation_creation(self):
        """Test creating Citation instance"""
        citation = Citation(
            query_id=1,
            paper_id=2,
            chunk_id=3,
            relevance_score=0.92
        )
        assert citation.query_id == 1
        assert citation.paper_id == 2
        assert citation.relevance_score == 0.92
    
    def test_citation_repr(self):
        """Test Citation string representation"""
        citation = Citation(
            query_id=1,
            paper_id=2,
            relevance_score=0.92
        )
        repr_str = repr(citation)
        assert "0.92" in repr_str
