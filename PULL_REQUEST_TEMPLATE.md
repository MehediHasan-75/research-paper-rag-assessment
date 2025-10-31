## 📝 Pull Request Template


```bash
- **Name**: John Doe
- **Email**: john.doe@email.com
- **LinkedIn**: linkedin.com/in/johndoe (optional)
- **Time Spent**: ~18 hours over 4 days
```
---

## 📝 Implementation Summary

I built a FastAPI-based RAG system using Qdrant for vector storage and Ollama (llama3) for generation. The system chunks PDFs intelligently, preserves section context, and provides cited answers with confidence scores. Key innovation: hierarchical chunking that maintains document structure for better retrieval.

---

## 🛠️ Technology Choices

**LLM**: [x] Ollama (model: llama3)  
**Why**: Local deployment, no API costs, good performance on technical content

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2  
**Why**: Fast, lightweight, good balance of speed vs accuracy for academic text

**Database**: PostgreSQL  
**Why**: Strong JSON support for metadata, familiar ecosystem, ACID compliance

**Key Libraries**:
- FastAPI - async support, auto-docs
- Qdrant-client - vector operations
- PyPDF2 - PDF extraction
- LangChain - RAG pipeline utilities

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB RAM minimum

### Quick Start (5 minutes)

1. **Clone and enter directory**
```bash
git clone https://github.com/YOUR_USERNAME/research-paper-rag-assessment.git
cd research-paper-rag-assessment
```

2. **Start services**
```bash
docker-compose up -d
# Starts Qdrant, PostgreSQL, and Ollama
```

3. **Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings (defaults work for docker-compose)
```

5. **Initialize database**
```bash
python src/init_db.py
```

6. **Run application**
```bash
uvicorn src.main:app --reload --port 8000
```

7. **Test it works**
```bash
# Upload a paper
curl -X POST "http://localhost:8000/api/papers/upload" \
  -F "file=@sample_papers/paper1_machine_learning.pdf"

# Query it
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used?"}'
```

8. **View API docs**: http://localhost:8000/docs

---

## 🏗️ Architecture Overview
`

**Key Components**:

1. **API Layer** (FastAPI/Any other)
   - Request validation
   - Error handling
   - Response formatting

2. **Processing Pipeline**
   - PDF text extraction
   - Hierarchical chunking (section-aware)
   - Embedding generation
   - Metadata enrichment

3. **Storage Layer**
   - Qdrant: 384-dim vectors, cosine similarity
   - PostgreSQL: Papers, queries, analytics

4. **RAG Pipeline**
   - Query understanding & expansion
   - Vector similarity search (top-5)
   - Context assembly (max 2000 tokens)
   - LLM generation with citations
   - Post-processing & validation

---

## 🎯 Design Decisions

### 1. Chunking Strategy
**Approach**: Hierarchical, section-aware chunking

- First, split by sections (Abstract, Introduction, etc.)
- Then, chunk each section with 500 token chunks, 50 token overlap
- Preserve section metadata in each chunk

**Rationale**: Academic papers have logical structure. Preserving this improves retrieval relevance by 30% in my tests.

**Trade-off**: More complex than simple splitting, but significantly better results.

### 2. Retrieval Method
**Approach**: Hybrid retrieval with re-ranking

1. Vector similarity search (top-10)
2. Re-rank by relevance score + metadata (section importance)
3. Return top-5 to LLM

**Rationale**: Pure vector search misses some relevant chunks. Re-ranking with section weights (Methods=1.2x, Results=1.1x) improves precision.

**Trade-off**: Slightly slower (~50ms extra) but worth it for accuracy.

### 3. Prompt Engineering
**Approach**: Structured prompt with XML tags

```
<context>
{retrieved_chunks}
</context>

<question>
{user_query}
</question>

<instructions>
Answer using ONLY the context. Include citations [Paper: X, Section: Y].
If uncertain, say so. Be concise.
</instructions>
```

**Rationale**: XML tags improve LLM instruction following. Citations enforce grounding.

---

## 🧪 Testing


**Test Results**:
- [x] All 5 papers ingested successfully (avg: 12 seconds each)
- [x] All API endpoints return proper status codes
- [x] 18/20 test queries return relevant answers
- [x] Citations properly formatted in 100% of responses
- [x] Error handling works for edge cases

---

## ✨ Bonus Features Implemented

- [x] **Docker Compose** - One-command setup
- [x] **Unit Tests** - 67% coverage, all core functions tested
- [ ] **Web UI** - Would add if more time
- [x] **Multi-paper Compare** - Works with paper_ids filter
- [x] **Caching** - Redis cache for embeddings (30% speedup)
- [x] **Analytics** - Track popular queries, avg response time