# src/models/database.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Index, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from src.config import settings

Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    authors = Column(JSON)  # List of author names
    year = Column(Integer, index=True)
    filename = Column(String(255), unique=True, nullable=False)
    file_path = Column(String(500))
    total_pages = Column(Integer)
    abstract = Column(Text)
    sections = Column(JSON)  # {section_name: page_range}
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False, index=True)  # ✅ Added index
    chunk_count = Column(Integer, default=0)
    
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")
    queries = relationship("QueryHistory", secondary="query_papers", back_populates="papers")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    chunk_index = Column(Integer)
    text = Column(Text, nullable=False)
    section = Column(String(500), index=True)  # Abstract, Introduction, etc.
    page_number = Column(Integer, index=True)
    vector_id = Column(String(100), unique=True, index=True)  # UUID for Qdrant
    embedding_generated = Column(Boolean, default=False)
    
    paper = relationship("Paper", back_populates="chunks")
    
    __table_args__ = (
        Index('idx_paper_section', 'paper_id', 'section'),  # ✅ Composite index
        Index('idx_paper_page', 'paper_id', 'page_number'),  # ✅ Composite index
    )

class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False, index=True)
    answer = Column(Text)
    top_k = Column(Integer, default=5)
    response_time = Column(Float)  # in seconds
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    paper_filter = Column(JSON)  # Optional paper_ids used
    user_rating = Column(Integer)  # 1-5 optional rating
    
    papers = relationship("Paper", secondary="query_papers", back_populates="queries")
    citations = relationship("Citation", back_populates="query", cascade="all, delete-orphan")

class Citation(Base):
    __tablename__ = "citations"
    
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)  # ✅ Added index
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), index=True)  # ✅ Added index
    relevance_score = Column(Float)
    
    query = relationship("QueryHistory", back_populates="citations")
    
# Association table for many-to-many relationship
class QueryPaper(Base):
    __tablename__ = "query_papers"
    
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), primary_key=True, index=True)  # ✅ Added index
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True, index=True)  # ✅ Added index

# ✅ DATABASE CONNECTION WITH POOLING (from env via settings)
DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables with indexes"""
    print("Creating database tables and indexes...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized with optimizations!")
