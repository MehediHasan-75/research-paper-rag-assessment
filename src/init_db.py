#!/usr/bin/env python3
"""
Initial Database Setup Script
Run this after cloning the repository to initialize the database with all tables and indexes.

Usage:
    python -m src.init_db              # Create tables only
    python -m src.init_db --reset      # Drop and recreate all tables
    python -m src.init_db --migrate    # Run migrations and updates
    python -m src.init_db --check      # Verify database integrity
    python -m src.init_db --seed       # Seed with sample data
    python -m src.init_db --all        # Do everything
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for proper module resolution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import (
    Base, engine, SessionLocal, Paper, Chunk, QueryHistory, Citation, 
    PaperArchive, init_db, check_database_integrity, migrate_to_enhanced_schema
)
from src.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def drop_tables():
    """Drop all tables (WARNING: destructive operation)."""
    logger.warning("=" * 60)
    logger.warning("WARNING: About to drop ALL tables!")
    logger.warning("=" * 60)
    
    confirmation = input("Type 'yes' to confirm: ").strip().lower()
    if confirmation != 'yes':
        logger.info("Operation cancelled.")
        return False
    
    try:
        logger.info("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("✓ All tables dropped successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to drop tables: {e}")
        return False


def reset_database():
    """Drop and recreate all tables."""
    logger.info("Resetting database...")
    
    if not drop_tables():
        return False
    
    try:
        init_db()
        logger.info("✓ Database reset complete!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to reset database: {e}")
        return False


def verify_tables():
    """Verify all tables exist."""
    logger.info("=" * 60)
    logger.info("Verifying database tables...")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        
        # Check each table
        tables_to_check = {
            'papers': Paper,
            'chunks': Chunk,
            'query_history': QueryHistory,
            'citations': Citation,
            'paper_archive': PaperArchive
        }
        
        for table_name, model in tables_to_check.items():
            try:
                count = db.query(model).count()
                logger.info(f"✓ Table '{table_name}': OK ({count} rows)")
            except Exception as e:
                logger.error(f"✗ Table '{table_name}': MISSING or ERROR - {e}")
                db.close()
                return False
        
        db.close()
        logger.info("✓ All tables verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def run_checks():
    """Run database integrity checks."""
    logger.info("=" * 60)
    logger.info("Checking database integrity...")
    logger.info("=" * 60)
    
    try:
        result = check_database_integrity()
        return result
    except Exception as e:
        logger.error(f"✗ Integrity check failed: {e}")
        return False


def run_migrations():
    """Run database migrations."""
    logger.info("=" * 60)
    logger.info("Running database migrations...")
    logger.info("=" * 60)
    
    try:
        migrate_to_enhanced_schema()
        logger.info("✓ Migrations completed successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Migration failed: {e}")
        return False


def seed_sample_data():
    """Seed database with sample data."""
    logger.info("=" * 60)
    logger.info("Seeding database with sample data...")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        
        # Check if sample data already exists
        existing_papers = db.query(Paper).filter(
            Paper.paper_name.like('sample_%')
        ).count()
        
        if existing_papers > 0:
            logger.info("Sample data already exists. Skipping seed.")
            db.close()
            return True
        
        # Create sample papers
        sample_papers = [
            Paper(
                paper_name="sample_lwe_crypto",
                title="Learning with Error: A Comprehensive Survey",
                authors=["Alice Smith", "Bob Johnson"],
                year=2024,
                filename="sample_lwe_2024.pdf",
                file_path="/uploads/sample_lwe_2024.pdf",
                total_pages=25,
                abstract="A comprehensive survey on Learning with Error cryptography...",
                quality_score=0.92,
                keywords=["LWE", "cryptography", "lattice"],
                format_type="standard",
                processed=True,
                chunk_count=15
            ),
            Paper(
                paper_name="sample_homomorphic_enc",
                title="Homomorphic Encryption: Theory and Practice",
                authors=["Charlie Brown", "Diana Prince"],
                year=2023,
                filename="sample_homomorphic_2023.pdf",
                file_path="/uploads/sample_homomorphic_2023.pdf",
                total_pages=30,
                abstract="Practical guide to homomorphic encryption schemes...",
                quality_score=0.88,
                keywords=["homomorphic", "encryption", "FHE"],
                format_type="standard",
                processed=True,
                chunk_count=20
            )
        ]
        
        for paper in sample_papers:
            db.add(paper)
        
        db.commit()
        
        # Create sample chunks
        paper1 = db.query(Paper).filter(
            Paper.paper_name == "sample_lwe_crypto"
        ).first()
        
        if paper1:
            sample_chunks = [
                Chunk(
                    paper_id=paper1.id,
                    chunk_index=0,
                    text="Learning with Error (LWE) is a computational problem that forms the basis of many modern cryptographic schemes...",
                    section="Introduction",
                    page_number=1,
                    section_id="1.0",
                    section_level=0
                ),
                Chunk(
                    paper_id=paper1.id,
                    chunk_index=1,
                    text="The LWE problem is defined as follows: Given samples from a distribution, recover the secret...",
                    section="Preliminaries",
                    page_number=2,
                    section_id="2.0",
                    section_level=0
                )
            ]
            
            for chunk in sample_chunks:
                db.add(chunk)
        
        db.commit()
        
        logger.info("✓ Sample data seeded successfully!")
        logger.info(f"  - Created {len(sample_papers)} sample papers")
        logger.info(f"  - Created sample chunks for demonstration")
        
        db.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Seed failed: {e}")
        db.close()
        return False


def print_summary():
    """Print database summary."""
    logger.info("=" * 60)
    logger.info("Database Summary")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        
        papers_count = db.query(Paper).count()
        chunks_count = db.query(Chunk).count()
        queries_count = db.query(QueryHistory).count()
        citations_count = db.query(Citation).count()
        
        logger.info(f"Papers: {papers_count}")
        logger.info(f"Chunks: {chunks_count}")
        logger.info(f"Queries: {queries_count}")
        logger.info(f"Citations: {citations_count}")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to retrieve summary: {e}")


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(
        description="Initialize Research Paper RAG Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  python -m src.init_db              # Create tables (default)
  python -m src.init_db --reset      # Drop and recreate tables
  python -m src.init_db --migrate    # Run migrations
  python -m src.init_db --check      # Check integrity
  python -m src.init_db --seed       # Seed sample data
  python -m src.init_db --verify     # Verify tables
  python -m src.init_db --all        # Do everything (create, migrate, check, seed)
        """
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop and recreate all tables'
    )
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Run database migrations'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check database integrity'
    )
    parser.add_argument(
        '--seed',
        action='store_true',
        help='Seed database with sample data'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Create tables, run migrations, check integrity, and seed'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify all tables exist'
    )
    
    args = parser.parse_args()
    
    logger.info("Research Paper RAG - Database Initialization")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    logger.info("")
    
    success = True
    
    # If --all specified, run all operations
    if args.all:
        args.reset = True
        args.migrate = True
        args.check = True
        args.seed = True
    
    # Default: create tables if no args
    if not any([args.reset, args.migrate, args.check, args.seed, args.verify]):
        logger.info("Running default initialization (create tables)...")
        try:
            init_db()
            success = success and verify_tables()
        except Exception as e:
            logger.error(f"✗ Failed to create tables: {e}")
            success = False
    else:
        # Reset database
        if args.reset:
            success = success and reset_database()
        
        # Create tables if not reset
        if not args.reset and (args.migrate or args.check or args.seed or args.verify):
            try:
                init_db()
                success = True
            except Exception as e:
                logger.error(f"✗ Failed to create tables: {e}")
                success = False
        
        # Verify tables
        if args.verify:
            success = success and verify_tables()
        
        # Run migrations
        if args.migrate:
            success = success and run_migrations()
        
        # Check integrity
        if args.check:
            success = success and run_checks()
        
        # Seed data
        if args.seed:
            success = success and seed_sample_data()
    
    # Print summary
    print_summary()
    
    logger.info("=" * 60)
    if success:
        logger.info("✓ Database initialization completed successfully!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("✗ Database initialization failed!")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
