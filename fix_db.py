#!/usr/bin/env python3
"""
reset_db_complete.py - Complete database reset from scratch
Drops and recreates the entire database with proper schema
"""

import subprocess
import sys
import time
from pathlib import Path

def run_sql(sql_commands, db_name="postgres"):
    """Execute SQL commands."""
    try:
        result = subprocess.run(
            ["psql", "-U", "rag_user", "-d", db_name, "-v", "ON_ERROR_STOP=1"],
            input=sql_commands,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 70)
    print("🗑️  COMPLETE DATABASE RESET")
    print("=" * 70)
    print("\n⚠️  WARNING: This will:")
    print("   - DROP the entire rag_db database")
    print("   - DELETE all papers, chunks, queries, embeddings")
    print("   - CREATE a fresh database with new schema")
    print()
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return False
    
    print("\n" + "=" * 70)
    
    # Step 1: Drop database
    print("\n1️⃣  Dropping existing database...")
    sql = "DROP DATABASE IF EXISTS rag_db;"
    success, stdout, stderr = run_sql(sql, "postgres")
    if success:
        print("   ✅ Database dropped")
    else:
        print(f"   ❌ Error: {stderr}")
        return False
    
    time.sleep(1)
    
    # Step 2: Create fresh database
    print("\n2️⃣  Creating fresh database...")
    sql = "CREATE DATABASE rag_db;"
    success, stdout, stderr = run_sql(sql, "postgres")
    if success:
        print("   ✅ Database created")
    else:
        print(f"   ❌ Error: {stderr}")
        return False
    
    time.sleep(1)
    
    # Step 3: Initialize schema
    print("\n3️⃣  Initializing database schema...")
    try:
        # Stop any running API first
        subprocess.run(
            ["pkill", "-f", "uvicorn"],
            capture_output=True,
            timeout=5
        )
        time.sleep(2)
        
        # Import and run database initialization
        from src.models.database import init_db, migrate_to_enhanced_schema, check_database_integrity
        
        print("   ✅ Importing database models...")
        
        # Initialize tables
        init_db()
        print("   ✅ Tables created")
        
        # Run migrations
        migrate_to_enhanced_schema()
        print("   ✅ Migrations applied")
        
        # Check integrity
        check_database_integrity()
        print("   ✅ Integrity verified")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Clear uploads
    print("\n4️⃣  Clearing uploads folder...")
    try:
        uploads_path = Path("uploads")
        if uploads_path.exists():
            import shutil
            shutil.rmtree(uploads_path)
        uploads_path.mkdir(parents=True, exist_ok=True)
        print("   ✅ Uploads folder cleared")
    except Exception as e:
        print(f"   ⚠️  Could not clear uploads: {e}")
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("✅ DATABASE RESET COMPLETE!")
    print("=" * 70)
    print("\n📝 Summary:")
    print("   ✅ Database: rag_db (fresh)")
    print("   ✅ Schema: Initialized with all tables")
    print("   ✅ Migrations: Applied (paper_name, quality_score, etc.)")
    print("   ✅ Uploads: Cleared")
    print()
    print("🚀 Next steps:")
    print("   1. Start API: uvicorn src.main:app --reload")
    print("   2. Upload papers: curl -X POST http://localhost:8000/api/papers/upload -F 'file=@paper.pdf'")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
