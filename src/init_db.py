#!/usr/bin/env python3
"""
Initial Database Setup Script
Run this after cloning the repository to initialize the database with all tables and indexes.

Usage:
    python initdb.py              # Create tables only
    python initdb.py --reset      # Drop and recreate all tables
    python initdb.py --migrate    # Run migrations and updates
    python initdb.py --check      # Verify database integrity
    python initdb.py --seed       # Seed with sample data
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import (
    Base, engine, SessionLocal, Paper, Chunk, QueryHistory, Citation, 
    PaperArchive, initdb, check_database_integrity, migrate_to_enhanced_schema
)
from src.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tables():
    """Create all database tables and indexes."""
    logger.info("=" * 60)
    logger.info("Creating database tables and indexes...")
    logger.info("=" * 60)
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ All tables created successfully!")
        logger.info(f"Database URL: {settings.DATABASEURL}")
        logger.info(f"Pool size: {settings.DBPOOLSIZE}")
        logger.info(f"Max overflow: {settings.DBMAXOVERFLOW}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create tables: {e}")
        return False


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
    
    if not create_tables():
        return False
    
    logger.info("✓ Database reset complete!")
    return True


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
            'queryhistory': QueryHistory,
            'citations': Citation,
            'paperarchive': PaperArchive
        }
        
        for table_name, model in tables_to_check.items():
            try:
                count = db.query(model).count()
                logger.info(f"✓ Table '{table_name}': OK (0 rows)")
            except Exception as e:
                logger.error(f"✗ Table '{table_name}': MISSING or ERROR - {e}")
                return False
        
        db.close()
        logger.info("✓ All tables verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def check_integrity():
    """Check database integrity."""
    logger.info("=" * 60)
    logger.info("Checking database integrity...")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        is_valid = check_database_integrity()
        db.close()
        
        if is_valid:
            logger.info("✓ Database integrity check passed!")
            return True
        else:
            logger.warning("✗ Database integrity issues detected!")
            return False
            
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
Examples:
  python initdb.py              # Create tables
  python initdb.py --reset      # Drop and recreate tables
  python initdb.py --migrate    # Run migrations
  python initdb.py --check      # Check integrity
  python initdb.py --seed       # Seed sample data
  python initdb.py --all        # Do everything (create, migrate, check, seed)
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
        logger.info("Running default initialization...")
        success = create_tables() and verify_tables()
    else:
        # Reset database
        if args.reset:
            success = success and reset_database()
        
        # Create tables if not reset
        if not args.reset and (args.migrate or args.check or args.seed or args.verify):
            success = success and create_tables()
        
        # Verify tables
        if args.verify:
            success = success and verify_tables()
        
        # Run migrations
        if args.migrate:
            success = success and run_migrations()
        
        # Check integrity
        if args.check:
            success = success and check_integrity()
        
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