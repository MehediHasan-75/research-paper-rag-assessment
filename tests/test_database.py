import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from datetime import datetime, timedelta
import json

from src.models.database import (
    Base, Paper, Chunk, QueryHistory, Citation, QueryPaper, SessionLocal, get_db
)

# ========== FIXTURES ==========

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    # Use SQLite for testing (in-memory)
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    return engine

@pytest.fixture
def test_db(test_engine):
    """Create test session"""
    TestSessionLocal = sessionmaker(bind=test_engine)
    db = TestSessionLocal()
    yield db
    db.close()

@pytest.fixture
def sample_paper(test_db: Session):
    """Create sample paper"""
    paper = Paper(
        title="Test Paper: Machine Learning",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        filename="test_paper.pdf",
        file_path="/papers/test_paper.pdf",
        total_pages=10,
        abstract="This is a test paper about ML.",
        sections={"Abstract": "p1", "Introduction": "p2", "Methods": "p3"},
        processed=True,
        chunk_count=5
    )
    test_db.add(paper)
    test_db.commit()
    test_db.refresh(paper)
    return paper

# ========== PAPER TABLE TESTS ==========

def test_paper_creation(test_db: Session):
    """Test basic paper creation"""
    paper = Paper(
        title="Test Paper",
        authors=["Author1"],
        year=2023,
        filename="test.pdf",
        processed=False
    )
    test_db.add(paper)
    test_db.commit()
    test_db.refresh(paper)
    
    assert paper.id is not None
    assert paper.title == "Test Paper"
    assert paper.upload_date is not Non
