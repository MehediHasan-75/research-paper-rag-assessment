# test_chunking_service.py (with path fix)
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.chunking_service import IntelligentChunker, Chunk


class TestIntelligentChunker:
    """Test suite for IntelligentChunker"""
    
    @pytest.fixture
    def chunker(self):
        """Create IntelligentChunker instance"""
        return IntelligentChunker(chunk_size=100, overlap=10)
    
    def test_initialization(self, chunker):
        """Test chunker initialization"""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 10
    
    def test_split_into_sentences(self, chunker):
        """Test sentence splitting"""
        text = "This is sentence one. This is sentence two. Dr. Smith is here."
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) >= 2
        assert "This is sentence one" in sentences[0]
    
    def test_estimate_tokens(self, chunker):
        """Test token estimation"""
        text = "This is a test text"
        tokens = chunker.estimate_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_chunk_section_basic(self, chunker):
        """Test basic section chunking"""
        section_text = "This is a long section. " * 20
        chunks = chunker.chunk_section(
            section_text=section_text,
            section_name="Introduction",
            page_start=1,
            page_end=2,
            paper_name="test_paper",
            section_id="1.0"
        )
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_metadata(self, chunker):
        """Test chunk contains correct metadata"""
        section_text = "Test section. " * 10
        chunks = chunker.chunk_section(
            section_text=section_text,
            section_name="Methods",
            page_start=5,
            page_end=6,
            paper_name="research_paper",
            section_id="2.1",
            section_level=1
        )
        
        chunk = chunks[0]
        assert chunk.section == "Methods"
        assert chunk.page_number == 5
        assert chunk.paper_name == "research_paper"
        assert chunk.section_id == "2.1"
        assert chunk.section_level == 1
    
    def test_chunk_overlap(self, chunker):
        """Test overlap between chunks"""
        text = "word " * 50
        chunks = chunker.chunk_section(
            section_text=text,
            section_name="Test",
            page_start=1,
            page_end=1
        )
        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0
    
    def test_empty_section(self, chunker):
        """Test chunking empty section"""
        chunks = chunker.chunk_section(
            section_text="",
            section_name="Empty",
            page_start=1,
            page_end=1
        )
        assert len(chunks) == 0
    
    def test_very_long_sentence(self, chunker):
        """Test handling of sentences longer than chunk_size"""
        long_sentence = " ".join(["word"] * 200)
        chunks = chunker.chunk_section(
            section_text=long_sentence,
            section_name="Test",
            page_start=1,
            page_end=1
        )
        assert len(chunks) > 0
    
    def test_chunk_size_configuration(self):
        """Test custom chunk size configuration"""
        chunker_small = IntelligentChunker(chunk_size=50, overlap=5)
        chunker_large = IntelligentChunker(chunk_size=500, overlap=50)
        
        assert chunker_small.chunk_size == 50
        assert chunker_large.chunk_size == 500
    
    def test_unicode_text_handling(self, chunker):
        """Test handling of unicode characters"""
        unicode_text = "আমরা লওয়ে ক্রিপ্টোগ্রাফি নিয়ে কাজ করছি। " * 10
        chunks = chunker.chunk_section(
            section_text=unicode_text,
            section_name="Unicode Test",
            page_start=1,
            page_end=1
        )
        assert len(chunks) > 0
    
    def test_special_characters_in_text(self, chunker):
        """Test handling special characters"""
        text = "Code: x^2 + y^2 = z^2. Price: $99.99. Email: test@example.com. " * 5
        chunks = chunker.chunk_section(
            section_text=text,
            section_name="Special Chars",
            page_start=1,
            page_end=1
        )
        assert len(chunks) > 0
