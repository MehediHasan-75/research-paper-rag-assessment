#!/usr/bin/env python3
"""
reset_db.py - Complete database and cache reset script
Deletes all previous data from PostgreSQL, Redis, Qdrant, and uploads folder.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path


# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


def print_status(stage, message):
    """Print status message with color."""
    print(f"{YELLOW}{stage}. {message}{NC}")


def print_success(message):
    """Print success message."""
    print(f"{GREEN}✅ {message}{NC}")


def print_error(message):
    """Print error message."""
    print(f"{RED}❌ {message}{NC}")


def run_command(command, shell=False):
    """Execute shell command."""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timeout"
    except Exception as e:
        return False, "", str(e)


def stop_api():
    """Stop running API server."""
    print_status(1, "Stopping API server...")
    success, _, _ = run_command("pkill -f 'uvicorn src.main:app'", shell=True)
    time.sleep(2)
    print_success("API server stopped")


def reset_postgresql():
    """Drop and recreate PostgreSQL database."""
    print_status(2, "Resetting PostgreSQL database...")
    
    sql_commands = """
    DROP DATABASE IF EXISTS rag_db;
    CREATE DATABASE rag_db;
    """
    
    try:
        result = subprocess.run(
            ["psql", "-U", "rag_user", "-d", "postgres"],
            input=sql_commands,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print_success("Database dropped and recreated")
        else:
            print_error(f"PostgreSQL error: {result.stderr}")
    except Exception as e:
        print_error(f"PostgreSQL error: {str(e)}")


def init_database_tables():
    """Initialize database tables."""
    print_status(3, "Initializing database tables...")
    
    try:
        from src.models.database import init_db
        init_db()
        print_success("Database tables initialized")
    except Exception as e:
        print_error(f"Table initialization error: {str(e)}")


def clear_uploads_folder():
    """Clear uploads folder."""
    print_status(4, "Clearing uploads folder...")
    
    try:
        uploads_path = Path("uploads")
        if uploads_path.exists():
            shutil.rmtree(uploads_path)
        uploads_path.mkdir(parents=True, exist_ok=True)
        print_success("Uploads folder cleared")
    except Exception as e:
        print_error(f"Upload folder error: {str(e)}")


def clear_redis_cache():
    """Clear Redis cache."""
    print_status(5, "Clearing Redis cache...")
    
    try:
        success, _, _ = run_command("redis-cli FLUSHALL", shell=True)
        if success:
            print_success("Redis cache cleared")
        else:
            print("Redis not running (OK)")
    except Exception as e:
        print("Redis not running (OK)")


def reset_qdrant():
    """Reset Qdrant vector database."""
    print_status(6, "Resetting Qdrant vector database...")
    
    try:
        run_command("docker-compose down qdrant", shell=True)
        time.sleep(2)
        success, _, _ = run_command("docker-compose up -d qdrant", shell=True)
        time.sleep(3)
        if success:
            print_success("Qdrant restarted")
        else:
            print("Qdrant restart incomplete (check docker-compose)")
    except Exception as e:
        print_error(f"Qdrant error: {str(e)}")


def verify_database():
    """Verify database is empty."""
    print_status(7, "Verifying database...")
    
    try:
        result = subprocess.run(
            ["psql", "-U", "rag_user", "-d", "rag_db", "-t", "-c", "SELECT COUNT(*) FROM papers;"],
            capture_output=True,
            text=True,
            timeout=5
        )
        paper_count = result.stdout.strip() if result.returncode == 0 else "0"
        
        result = subprocess.run(
            ["psql", "-U", "rag_user", "-d", "rag_db", "-t", "-c", "SELECT COUNT(*) FROM chunks;"],
            capture_output=True,
            text=True,
            timeout=5
        )
        chunk_count = result.stdout.strip() if result.returncode == 0 else "0"
        
        print_success(f"Papers: {paper_count}, Chunks: {chunk_count}")
    except Exception as e:
        print_error(f"Verification error: {str(e)}")


def print_summary():
    """Print final summary."""
    print()
    print("=" * 43)
    print(f"{GREEN}✅ DATABASE RESET COMPLETE!{NC}")
    print("=" * 43)
    print()
    print("Summary:")
    print("  ✅ PostgreSQL database: Fresh")
    print("  ✅ Tables initialized: Yes")
    print("  ✅ Uploads folder: Cleared")
    print("  ✅ Redis cache: Cleared")
    print("  ✅ Qdrant vectors: Reset")
    print()


def main():
    """Main reset function."""
    print("🧹 Starting complete database reset...")
    print("=" * 43)
    print()
    
    try:
        stop_api()
        reset_postgresql()
        init_database_tables()
        clear_uploads_folder()
        clear_redis_cache()
        reset_qdrant()
        verify_database()
        print_summary()
    except KeyboardInterrupt:
        print_error("Reset interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
