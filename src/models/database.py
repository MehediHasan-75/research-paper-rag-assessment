from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Index, Boolean, create_engine, UniqueConstraint, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, validates
from datetime import datetime
import os
from src.config import settings


Base = declarative_base()


class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # ✅ Paper name from PDF filename (stable identifier)
    paper_name = Column(
        String(255), 
        unique=True, 
        nullable=False, 
        index=True
    )
    
    title = Column(String(500), nullable=False, index=True)
    authors = Column(JSON, default=[])  # List of author names
    year = Column(Integer, index=True, nullable=True)  # ✅ NULLABLE - handle invalid years
    filename = Column(String(255), unique=True, nullable=False)
    file_path = Column(String(500))
    total_pages = Column(Integer)
    abstract = Column(Text)
    sections = Column(JSON, default={})  # {section_name: page_range}
    
    # ✅ Quality metrics
    quality_score = Column(
        Float, 
        default=0.0,
        index=True
    )
    keywords = Column(JSON, default=[])  # Extracted keywords
    format_type = Column(String(50), default="standard")  # standard/report/commentary
    
    # Timestamps
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing status
    processed = Column(Boolean, default=False, index=True)
    chunk_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")
    citations = relationship("Citation", back_populates="paper", cascade="all, delete-orphan")
    
    # ✅ FIXED: Better validation for quality_score
    @validates('quality_score')
    def validate_quality_score(self, key, value):
        if value is None:
            return 0.0
        if not 0 <= value <= 1:
            raise ValueError("quality_score must be between 0 and 1")
        return value
    
    # ✅ FIXED: Better validation for year (more lenient)
    @validates('year')
    def validate_year(self, key, value):
        if value is None:
            return None  # Allow NULL years
        
        # Convert to int if it's a string
        if isinstance(value, str):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return None  # Return None if can't convert
        
        # Check if year is reasonable
        current_year = datetime.now().year
        
        # Valid range: 1000 to 2100 (more lenient)
        if isinstance(value, int):
            if value <= 0 or value > 2150:  # ✅ Allow up to 2150
                # Log warning but don't fail
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️  Invalid year detected: {value}. Setting to None.")
                return None  # Return None instead of raising error
        
        return value

    def __repr__(self):
        return f"<Paper(id={self.id}, paper_name={self.paper_name}, quality={self.quality_score:.2f})>"


class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    chunk_index = Column(Integer)
    text = Column(Text, nullable=False)
    section = Column(String(500), index=True)  # Abstract, Introduction, etc.
    page_number = Column(Integer, index=True)
    
    # ✅ Section hierarchy fields
    section_id = Column(
        String(50), 
        nullable=True,
        index=True  # Fast section-based search
    )
    section_level = Column(Integer, default=0)  # 0=main, 1=sub, 2=subsub
    
    # Vector storage
    vector_id = Column(String(255), unique=True, index=True)  # UUID for Qdrant
    embedding_generated = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    paper = relationship("Paper", back_populates="chunks")
    
    __table_args__ = (
        Index('idx_paper_chunk', 'paper_id', 'chunk_index'),  # Composite for efficient lookup
        Index('idx_paper_section', 'paper_id', 'section_id'),  # Composite for section search
        Index('idx_paper_page', 'paper_id', 'page_number'),  # Composite for page queries
        CheckConstraint(
            "(embedding_generated = FALSE) OR (vector_id IS NOT NULL)",
            name='ck_vector_id_when_generated'  # Prevent orphaned embeddings
        ),
    )
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, paper_id={self.paper_id}, section={self.section_id})>"


class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False, index=True)
    answer = Column(Text)
    top_k = Column(Integer, default=5)
    response_time = Column(Float)  # in seconds
    confidence = Column(Float)  # 0-1 confidence score
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    paper_filter = Column(JSON)  # Optional paper_ids used
    user_rating = Column(Integer)  # 1-5 optional rating
    
    # ✅ Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    citations = relationship("Citation", back_populates="query", cascade="all, delete-orphan")
    
    # ✅ Composite indexes for analytics
    __table_args__ = (
        Index('idx_confidence', 'confidence'),  # Filter by confidence
        Index('idx_user_rating', 'user_rating'),  # Filter by rating
        Index('idx_created_at_query', 'created_at'),  # Time-range queries
    )
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, query={self.query_text[:40]}..., confidence={self.confidence:.2f})>"


class Citation(Base):
    __tablename__ = "citations"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True, index=True)
    relevance_score = Column(Float, default=0.0)
    
    # ✅ Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    query = relationship("QueryHistory", back_populates="citations")
    paper = relationship("Paper", back_populates="citations")
    
    # ✅ Unique constraint to prevent duplicate citations
    __table_args__ = (
        UniqueConstraint('query_id', 'paper_id', name='uq_query_paper'),
    )
    
    def __repr__(self):
        return f"<Citation(query_id={self.query_id}, paper_id={self.paper_id}, score={self.relevance_score:.2f})>"


# ✅ OPTIONAL: Association table (kept for reference, not actively used in relationships)
class QueryPaper(Base):
    __tablename__ = "query_papers"
    
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True, index=True)
    
    def __repr__(self):
        return f"<QueryPaper(query_id={self.query_id}, paper_id={self.paper_id})>"


# ✅ Archive table for soft deletes
class PaperArchive(Base):
    __tablename__ = "paper_archive"
    
    id = Column(Integer, primary_key=True, index=True)
    original_paper_id = Column(Integer, nullable=False)
    paper_name = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    archived_at = Column(DateTime, default=datetime.utcnow, index=True)
    archived_reason = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<PaperArchive(paper_name={self.paper_name}, archived_at={self.archived_at})>"


# ========== DATABASE CONNECTION WITH POOLING ==========

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


# ========== DATABASE INTEGRITY & MIGRATION ==========

def check_database_integrity():
    """Verify database integrity and fix issues"""
    from sqlalchemy import text, func
    db = SessionLocal()
    issues = []
    
    try:
        # Check 1: Papers without vector IDs but with embedding_generated=True
        orphaned_chunks = db.query(Chunk).filter(
            Chunk.embedding_generated == True,
            Chunk.vector_id == None
        ).count()
        if orphaned_chunks > 0:
            issues.append(f"Found {orphaned_chunks} chunks with orphaned vectors")
        
        # Check 2: Citations referencing non-existent papers
        missing_citations = db.query(Citation).filter(
            ~Citation.paper_id.in_(db.query(Paper.id))
        ).count()
        if missing_citations > 0:
            issues.append(f"Found {missing_citations} citations with missing papers")
        
        # Check 3: Quality scores out of range
        bad_scores = db.query(Paper).filter(
            (Paper.quality_score < 0) | (Paper.quality_score > 1)
        ).count()
        if bad_scores > 0:
            issues.append(f"Found {bad_scores} papers with invalid quality scores")
        
        # Check 4: Duplicate papers by paper_name
        duplicate_papers = db.query(
            Paper.paper_name,
            func.count(Paper.id).label('count')
        ).group_by(Paper.paper_name).having(
            func.count(Paper.id) > 1
        ).count()
        if duplicate_papers > 0:
            issues.append(f"Found {duplicate_papers} duplicate paper_names")
        
        if issues:
            print("⚠️  Database integrity issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Database integrity check passed")
            return True
    
    finally:
        db.close()


def migrate_to_enhanced_schema():
    """
    Run migrations to add new columns if they don't exist
    Use this for upgrading from old schema
    """
    from sqlalchemy import text
    
    print("Running database migrations...")
    
    migrations = [
        # Add paper_name if missing
        "ALTER TABLE papers ADD COLUMN paper_name VARCHAR(255) UNIQUE",
        
        # Add quality metrics if missing
        "ALTER TABLE papers ADD COLUMN quality_score FLOAT DEFAULT 0.0",
        "ALTER TABLE papers ADD COLUMN keywords JSONB DEFAULT '[]'",
        "ALTER TABLE papers ADD COLUMN format_type VARCHAR(50) DEFAULT 'standard'",
        
        # Add section hierarchy if missing
        "ALTER TABLE chunks ADD COLUMN section_id VARCHAR(50)",
        "ALTER TABLE chunks ADD COLUMN section_level INTEGER DEFAULT 0",
        
        # Add timestamps if missing
        "ALTER TABLE papers ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE papers ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE chunks ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE query_history ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE query_history ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE citations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        
        # Create indexes
        "CREATE INDEX IF NOT EXISTS idx_paper_name ON papers(paper_name)",
        "CREATE INDEX IF NOT EXISTS idx_quality_score ON papers(quality_score)",
        "CREATE INDEX IF NOT EXISTS idx_section_id ON chunks(section_id)",
        "CREATE INDEX IF NOT EXISTS idx_paper_chunk ON chunks(paper_id, chunk_index)",
        "CREATE INDEX IF NOT EXISTS idx_confidence ON query_history(confidence)",
        "CREATE INDEX IF NOT EXISTS idx_created_at_query ON query_history(created_at)",
    ]
    
    with engine.connect() as conn:
        for migration in migrations:
            try:
                conn.execute(text(migration))
                print(f"✅ {migration[:60]}...")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"⊘ Already exists: {migration[:60]}...")
                else:
                    print(f"❌ Migration failed: {migration[:60]}...")
                    print(f"   Error: {str(e)[:100]}")
        
        conn.commit()
    
    print("✅ Database migrations complete")


if __name__ == "__main__":
    # Initialize database on first run
    init_db()
    
    # Check integrity
    check_database_integrity()
    
    # Run migrations if needed
    try:
        migrate_to_enhanced_schema()
    except Exception as e:
        print(f"Migrations may have already been applied: {e}")
