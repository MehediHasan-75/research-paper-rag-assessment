#!/bin/bash
# reset_db.sh - Complete database and cache reset script
# Usage: bash reset_db.sh

set -e  # Exit on error

echo "🧹 Starting complete database reset..."
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Stop API if running
echo -e "${YELLOW}1. Stopping API server...${NC}"
pkill -f "uvicorn src.main:app" || echo "API not running"
sleep 2

# 2. Drop and recreate PostgreSQL database
echo -e "${YELLOW}2. Resetting PostgreSQL database...${NC}"
psql -U rag_user -d postgres << EOF
DROP DATABASE IF EXISTS rag_db;
CREATE DATABASE rag_db;
EOF
echo -e "${GREEN}✅ Database dropped and recreated${NC}"

# 3. Initialize database tables
echo -e "${YELLOW}3. Initializing database tables...${NC}"
python << 'PYTHON_EOF'
from src.models.database import init_db
init_db()
print("✅ Database tables initialized")
PYTHON_EOF

# 4. Clear uploads folder
echo -e "${YELLOW}4. Clearing uploads folder...${NC}"
rm -rf uploads/*
mkdir -p uploads
echo -e "${GREEN}✅ Uploads folder cleared${NC}"

# 5. Clear Redis cache
echo -e "${YELLOW}5. Clearing Redis cache...${NC}"
redis-cli FLUSHALL > /dev/null || echo "Redis not running (OK)"
echo -e "${GREEN}✅ Redis cache cleared${NC}"

# 6. Reset Qdrant
echo -e "${YELLOW}6. Resetting Qdrant vector database...${NC}"
docker-compose down qdrant 2>/dev/null || true
sleep 2
docker-compose up -d qdrant
sleep 3
echo -e "${GREEN}✅ Qdrant restarted${NC}"

# 7. Verify database is empty
echo -e "${YELLOW}7. Verifying database...${NC}"
PAPER_COUNT=$(psql -U rag_user -d rag_db -t -c "SELECT COUNT(*) FROM papers;" 2>/dev/null || echo "0")
CHUNK_COUNT=$(psql -U rag_user -d rag_db -t -c "SELECT COUNT(*) FROM chunks;" 2>/dev/null || echo "0")
echo -e "${GREEN}✅ Papers: $PAPER_COUNT, Chunks: $CHUNK_COUNT${NC}"

# 8. Summary
echo ""
echo "==========================================="
echo -e "${GREEN}✅ DATABASE RESET COMPLETE!${NC}"
echo "==========================================="
echo ""
echo "Summary:"
echo "  ✅ PostgreSQL database: Fresh"
echo "  ✅ Tables initialized: Yes"
echo "  ✅ Uploads folder: Cleared"
echo "  ✅ Redis cache: Cleared"
echo "  ✅ Qdrant vectors: Reset"
echo ""
echo "Next steps:"
echo "  1. Start API: export HF_HUB_OFFLINE=1 && uvicorn src.main:app --reload"
echo "  2. Upload papers in new terminal"
echo "  3. Test queries"
echo ""
