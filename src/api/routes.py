# src/api/routes.py (CORRECTED SECTION)
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
import shutil
import logging
from datetime import datetime

from src.models.database import Paper, Chunk, QueryHistory, Citation, QueryPaper, get_db
from src.services.pdf_processor import pdf_processor
from src.services.chunking_service import intelligent_chunker
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service
from src.services.cache_service import cache_service
from src.services.rag_pipeline import RAGPipeline
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

# Create uploads directory from settings
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


# ========== HELPER FUNCTIONS ==========

def get_rag_pipeline(db: Session) -> RAGPipeline:
    """Factory function to instantiate RAGPipeline with dependencies"""
    return RAGPipeline(
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        db_session=db,
        model=getattr(settings, 'LLM_MODEL', 'deepseek-r1:8b')
    )


def _save_query_history(
    db: Session,
    query_text: str,
    top_k: int,
    response_time: float,
    confidence: float,
    answer: str = None,
    paper_filter: Optional[List[int]] = None,
    user_rating: Optional[int] = None
) -> int:
    """
    Helper to save query to history for analytics
    Matches QueryHistory model exactly
    """
    query_history = QueryHistory(
        query_text=query_text,
        answer=answer,
        top_k=top_k,
        response_time=response_time,
        confidence=confidence,
        paper_filter=paper_filter,  # ✅ CORRECT FIELD NAME
        user_rating=user_rating
    )
    db.add(query_history)
    db.commit()
    db.refresh(query_history)
    logger.info(f"✅ Query saved to history (ID={query_history.id})")
    return query_history.id


def _save_citations(
    db: Session,
    query_id: int,
    rag_result: dict,
    paper_title_to_id_map: dict
):
    """
    Helper to save citations from RAG result to database
    
    Args:
        db: Database session
        query_id: Query history ID to link citations to
        rag_result: Result from RAGPipeline.generate_answer()
        paper_title_to_id_map: Dict mapping paper titles to IDs for quick lookup
    """
    for citation_data in rag_result.get('citations', []):
        paper_title = citation_data.get('paper_title')
        
        # Look up paper ID
        paper_id = paper_title_to_id_map.get(paper_title)
        if not paper_id:
            logger.warning(f"⚠️ Paper not found for citation: {paper_title}")
            continue
        
        try:
            citation = Citation(
                query_id=query_id,
                paper_id=paper_id,
                chunk_id=None,  # Optional: set if available in search results
                relevance_score=citation_data.get('relevance_score', 0.0)
            )
            db.add(citation)
        except Exception as e:
            logger.error(f"❌ Failed to save citation: {e}")
    
    db.commit()
    logger.info(f"✅ Saved {len(rag_result.get('citations', []))} citations")


# ========== DOCUMENT INGESTION SYSTEM ==========

@router.post("/papers/upload")
async def upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process PDF"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    if file.size and file.size > settings.MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    logger.info(f"📄 Uploading: {file.filename}")
    
    # Check if paper already exists
    existing_paper = db.query(Paper).filter(Paper.filename == file.filename).first()
    if existing_paper:
        logger.warning(f"⚠️ Paper already exists: {file.filename} (ID={existing_paper.id})")
        return {
            "paper_id": existing_paper.id,
            "title": existing_paper.title,
            "status": "already_exists",
            "message": "This paper was already uploaded"
        }
    
    file_path = os.path.join(settings.UPLOAD_DIR, str(uuid.uuid4()) + ".pdf")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # Process PDF
    processed_doc = pdf_processor.process_document(file_path)
    if not processed_doc:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail="PDF processing failed")
    
    # Save paper
    paper = Paper(
        title=processed_doc.title,
        authors=processed_doc.authors,
        year=processed_doc.year,
        filename=file.filename,
        file_path=file_path,
        total_pages=processed_doc.total_pages,
        abstract=processed_doc.abstract,
        sections={s.name: f"p{s.page_start}-{s.page_end}" for s in processed_doc.sections},
        processed=False
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    logger.info(f"✅ Paper saved: ID={paper.id}")
    
    # Step 1: Create chunks with BATCH COMMITS
    chunk_count = 0
    for section in processed_doc.sections:
        chunks = intelligent_chunker.chunk_section(
            section_text=section.text,
            section_name=section.name,
            page_start=section.page_start,
            page_end=section.page_end
        )
        
        for chunk in chunks:
            db_chunk = Chunk(
                paper_id=paper.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                section=chunk.section,
                page_number=chunk.page_number
            )
            db.add(db_chunk)
            chunk_count += 1
            
            # BATCH COMMIT: Every 10 chunks
            if chunk_count % 10 == 0:
                db.commit()
                logger.info(f"✅ Committed {chunk_count} chunks...")
    
    db.commit()
    logger.info(f"✅ All {chunk_count} chunks committed")
    
    # Step 2: Get chunk IDs from database
    db_chunks = db.query(Chunk).filter(Chunk.paper_id == paper.id).all()
    logger.info(f"✅ Retrieved {len(db_chunks)} chunks from database")
    
    chunk_dicts = [
        {
            'id': c.id,
            'text': c.text,
            'section': c.section,
            'page_number': c.page_number
        }
        for c in db_chunks
    ]
    
    # Step 3: Generate embeddings
    try:
        embedding_texts = [c['text'] for c in chunk_dicts]
        embeddings = embedding_service.encode_batch(embedding_texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")
    
    # Step 4: Store in Qdrant
    try:
        vector_ids = qdrant_service.upsert_chunks(chunk_dicts, embeddings, paper.id)
        logger.info(f"✅ Stored {len(vector_ids)} vectors in Qdrant")
    except Exception as e:
        logger.error(f"❌ Qdrant storage failed: {e}")
        raise HTTPException(status_code=500, detail="Vector storage failed")
    
    # Step 5: Update vector IDs
    for db_chunk, vector_id in zip(db_chunks, vector_ids):
        db_chunk.vector_id = vector_id
        db_chunk.embedding_generated = True
    
    db.commit()
    
    # Step 6: Mark as processed
    paper.processed = True
    paper.chunk_count = len(db_chunks)
    db.commit()
    
    logger.info(f"✅ Upload complete: {len(db_chunks)} chunks")
    
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "sections": [s.name for s in processed_doc.sections],
        "chunks_created": len(db_chunks),
        "pages": paper.total_pages,
        "upload_status": "success"
    }


# ========== RAG QUERY SYSTEM ==========

@router.post("/query")
async def rag_query(
    question: str = Query(..., min_length=3, max_length=500, description="User question"),
    top_k: int = Query(5, ge=1, le=20, description="Number of chunks to retrieve"),
    paper_ids: Optional[List[int]] = Query(None, description="Optional: filter by paper IDs"),
    use_cache: bool = Query(True, description="Enable response caching"),
    db: Session = Depends(get_db)
):
    """
    🚀 Full RAG-Powered Query Endpoint
    
    Uses semantic search + LLM generation for comprehensive answers.
    
    Request:
    POST /api/query?question=How does LWE enable homomorphic encryption?&top_k=5
    
    Response:
    {
        "answer": "LWE enables homomorphic encryption by...",
        "citations": [
            {
                "paper_title": "Learning with Errors",
                "section": "Applications",
                "page": 12,
                "relevance_score": 0.95
            }
        ],
        "sources_used": ["Paper Title 1", "Paper Title 2"],
        "confidence": 0.92,
        "response_time": 2.34,
        "cached": false
    }
    """
    logger.info(f"🔍 RAG Query: {question[:50]}...")
    
    # Check cache first
    if use_cache:
        cached = cache_service.get_query_cache(question, paper_ids)
        if cached:
            logger.info("✅ Cache hit")
            cached['cached'] = True
            return cached
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = get_rag_pipeline(db)
        
        # Generate answer using RAG
        rag_result = rag_pipeline.generate_answer(
            question=question,
            top_k=top_k,
            paper_ids=paper_ids
        )
        
        # Create paper title to ID mapping for citations
        paper_title_to_id_map = {}
        for citation_data in rag_result.get('citations', []):
            paper_title = citation_data.get('paper_title')
            paper = db.query(Paper).filter(Paper.title == paper_title).first()
            if paper:
                paper_title_to_id_map[paper_title] = paper.id
        
        # Save to query history for analytics
        query_id = _save_query_history(
            db=db,
            query_text=question,
            top_k=top_k,
            response_time=rag_result['response_time'],
            confidence=rag_result['confidence'],
            answer=rag_result['answer'],
            paper_filter=paper_ids
        )
        
        # Store citations in database
        _save_citations(db, query_id, rag_result, paper_title_to_id_map)
        
        # Cache result
        if use_cache:
            cache_service.set_query_cache(question, rag_result, paper_ids)
        
        rag_result['cached'] = False
        logger.info(f"✅ Query answered in {rag_result['response_time']}s")
        
        return rag_result
        
    except Exception as e:
        logger.error(f"❌ RAG query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query/advanced")
async def rag_query_advanced(
    question: str = Body(..., min_length=3, max_length=500),
    top_k: int = Body(5, ge=1, le=20),
    paper_ids: Optional[List[int]] = Body(None),
    use_cache: bool = Body(True),
    model: Optional[str] = Body(None, description="Override LLM model"),
    user_rating: Optional[int] = Body(None, ge=1, le=5, description="Rate the answer 1-5"),
    db: Session = Depends(get_db)
):
    """
    📚 Advanced RAG Query with Custom Configuration
    
    POST /api/query/advanced
    {
        "question": "What are the security implications of LWE?",
        "top_k": 10,
        "paper_ids": [1, 2, 3],
        "use_cache": true,
        "model": "deepseek-r1:8b",
        "user_rating": 5
    }
    """
    logger.info(f"🔬 Advanced RAG Query: {question[:50]}...")
    
    try:
        # Initialize RAG with custom model if specified
        rag_pipeline = get_rag_pipeline(db)
        if model:
            rag_pipeline.model = model
            logger.info(f"📝 Using model: {model}")
        
        # Check cache
        if use_cache:
            cached = cache_service.get_query_cache(question, paper_ids)
            if cached:
                cached['cached'] = True
                logger.info("✅ Cache hit")
                return cached
        
        # Generate answer
        rag_result = rag_pipeline.generate_answer(
            question=question,
            top_k=top_k,
            paper_ids=paper_ids
        )
        
        # Create paper title to ID mapping
        paper_title_to_id_map = {}
        for citation_data in rag_result.get('citations', []):
            paper_title = citation_data.get('paper_title')
            paper = db.query(Paper).filter(Paper.title == paper_title).first()
            if paper:
                paper_title_to_id_map[paper_title] = paper.id
        
        # Save to history
        query_id = _save_query_history(
            db=db,
            query_text=question,
            top_k=top_k,
            response_time=rag_result['response_time'],
            confidence=rag_result['confidence'],
            answer=rag_result['answer'],
            paper_filter=paper_ids,
            user_rating=user_rating
        )
        
        # Store citations
        _save_citations(db, query_id, rag_result, paper_title_to_id_map)
        
        # Cache
        if use_cache:
            cache_service.set_query_cache(question, rag_result, paper_ids)
        
        rag_result['cached'] = False
        rag_result['model_used'] = rag_pipeline.model
        rag_result['user_rating'] = user_rating
        
        return rag_result
        
    except Exception as e:
        logger.error(f"❌ Advanced query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Advanced query failed: {str(e)}")


@router.post("/query/batch")
async def rag_query_batch(
    questions: List[str] = Body(..., min_items=1, max_items=10),
    top_k: int = Body(5, ge=1, le=20),
    paper_ids: Optional[List[int]] = Body(None),
    db: Session = Depends(get_db)
):
    """
    ⚡ Batch RAG Query Processing
    
    POST /api/query/batch
    {
        "questions": [
            "What is LWE?",
            "How does homomorphic encryption work?",
            "What are practical applications?"
        ],
        "top_k": 5,
        "paper_ids": [1, 2]
    }
    """
    logger.info(f"⚡ Processing batch of {len(questions)} queries")
    
    results = []
    rag_pipeline = get_rag_pipeline(db)
    
    # Create paper title to ID mapping once for all queries
    paper_title_to_id_map = {}
    all_papers = db.query(Paper).all()
    for paper in all_papers:
        paper_title_to_id_map[paper.title] = paper.id
    
    for idx, question in enumerate(questions, 1):
        try:
            logger.info(f"Processing query {idx}/{len(questions)}")
            
            result = rag_pipeline.generate_answer(
                question=question,
                top_k=top_k,
                paper_ids=paper_ids
            )
            
            # Save to history
            query_id = _save_query_history(
                db=db,
                query_text=question,
                top_k=top_k,
                response_time=result['response_time'],
                confidence=result['confidence'],
                answer=result['answer'],
                paper_filter=paper_ids
            )
            
            # Save citations
            _save_citations(db, query_id, result, paper_title_to_id_map)
            
            result['question'] = question
            result['batch_index'] = idx
            result['success'] = True
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process query {idx}: {str(e)}")
            results.append({
                "question": question,
                "batch_index": idx,
                "error": str(e),
                "success": False
            })
    
    logger.info(f"✅ Batch processing complete: {len(results)} results")
    
    return {
        "total_queries": len(questions),
        "successful": sum(1 for r in results if r.get('success', True)),
        "results": results
    }


# ========== PAPER MANAGEMENT ==========

@router.get("/papers")
async def list_papers(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all uploaded papers with pagination"""
    papers = db.query(Paper).offset(skip).limit(limit).all()
    total = db.query(Paper).count()
    
    return {
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "pages": p.total_pages,
                "chunks": p.chunk_count,
                "processed": p.processed,
                "uploaded": p.upload_date.isoformat()
            }
            for p in papers
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """Get detailed paper information"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "sections": paper.sections,
        "pages": paper.total_pages,
        "chunks": paper.chunk_count,
        "processed": paper.processed,
        "uploaded": paper.upload_date.isoformat()
    }


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    """Remove paper + vectors from Qdrant + chunks from database"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Delete from Qdrant
    try:
        qdrant_service.delete_by_paper(paper_id)
    except Exception as e:
        logger.warning(f"Qdrant deletion failed: {e}")
    
    # Delete from database (cascades to chunks and citations)
    db.delete(paper)
    db.commit()
    
    logger.info(f"✅ Paper deleted: {paper.title}")
    
    return {
        "id": paper_id,
        "status": "deleted",
        "title": paper.title
    }


@router.get("/papers/{paper_id}/stats")
async def paper_stats(paper_id: int, db: Session = Depends(get_db)):
    """View paper statistics"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Count citations
    citations_count = db.query(Citation).filter(Citation.paper_id == paper_id).count()
    
    # Count queries that referenced this paper
    from sqlalchemy import func
    query_count = db.query(func.count(QueryHistory.id)).filter(
        QueryHistory.paper_filter.contains(str(paper_id))
    ).scalar() or 0
    
    return {
        "paper_id": paper_id,
        "title": paper.title,
        "chunks_created": paper.chunk_count,
        "times_cited": citations_count,
        "times_queried": query_count,
        "pages": paper.total_pages,
        "uploaded": paper.upload_date.isoformat()
    }


# ========== QUERY HISTORY & ANALYTICS ==========

@router.get("/queries/history")
async def queries_history(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Recent queries with response times"""
    queries = db.query(QueryHistory).order_by(
        QueryHistory.timestamp.desc()
    ).limit(limit).all()
    
    return {
        "queries": [
            {
                "id": q.id,
                "query": q.query_text,
                "answer_preview": q.answer[:200] if q.answer else None,
                "response_time_ms": round(q.response_time * 1000, 2) if q.response_time else None,
                "confidence": q.confidence,
                "user_rating": q.user_rating,
                "paper_filter": q.paper_filter,
                "timestamp": q.timestamp.isoformat()
            }
            for q in queries
        ],
        "total": len(queries)
    }


@router.get("/analytics/popular")
async def analytics_popular(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Most frequently asked questions"""
    from sqlalchemy import func
    
    popular = db.query(
        QueryHistory.query_text,
        func.count(QueryHistory.id).label('count'),
        func.avg(QueryHistory.confidence).label('avg_confidence')
    ).group_by(QueryHistory.query_text).order_by(
        func.count(QueryHistory.id).desc()
    ).limit(limit).all()
    
    return {
        "popular_queries": [
            {
                "query": q[0],
                "count": q[1],
                "avg_confidence": round(q[2], 3) if q[2] else 0
            }
            for q in popular
        ],
        "total_queries": db.query(QueryHistory).count()
    }


@router.get("/analytics/papers")
async def analytics_papers(db: Session = Depends(get_db)):
    """Paper usage statistics"""
    from sqlalchemy import func
    
    stats = db.query(
        Paper.id,
        Paper.title,
        func.count(Citation.id).label('citations')
    ).outerjoin(Citation, Paper.id == Citation.paper_id).group_by(
        Paper.id
    ).all()
    
    return {
        "paper_stats": [
            {
                "id": s[0],
                "title": s[1],
                "times_cited": s[2] or 0
            }
            for s in stats
        ]
    }


@router.get("/analytics/performance")
async def analytics_performance(db: Session = Depends(get_db)):
    """System performance metrics"""
    from sqlalchemy import func
    
    queries = db.query(QueryHistory).all()
    
    if not queries:
        return {
            "total_queries": 0,
            "avg_response_time_ms": 0,
            "avg_confidence": 0,
            "avg_user_rating": None
        }
    
    avg_time = sum(q.response_time or 0 for q in queries) / len(queries)
    avg_conf = sum(q.confidence or 0 for q in queries) / len(queries)
    
    # Calculate average user rating
    rated_queries = [q for q in queries if q.user_rating is not None]
    avg_rating = sum(q.user_rating for q in rated_queries) / len(rated_queries) if rated_queries else None
    
    return {
        "total_queries": len(queries),
        "total_papers": db.query(Paper).count(),
        "total_chunks": db.query(Chunk).count(),
        "avg_response_time_ms": round(avg_time * 1000, 2),
        "avg_confidence": round(avg_conf, 3),
        "avg_user_rating": round(avg_rating, 2) if avg_rating else None,
        "rated_responses_count": len(rated_queries)
    }


# ========== HEALTH CHECK & INFO ==========

@router.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "pdf_processor": "ready",
            "chunking": "ready",
            "embedding": "ready",
            "qdrant": "ready",
            "cache": "ready",
            "rag_pipeline": "ready"
        }
    }


@router.get("/")
async def root():
    """Welcome message with API documentation"""
    return {
        "message": "🚀 Research Paper RAG System",
        "version": "2.0",
        "endpoints": {
            "document_ingestion": {
                "upload_paper": "POST /api/papers/upload"
            },
            "rag_queries": {
                "basic_rag_query": "POST /api/query (with query params)",
                "advanced_rag_query": "POST /api/query/advanced (with body + user rating)",
                "batch_rag_query": "POST /api/query/batch"
            },
            "paper_management": {
                "list_papers": "GET /api/papers",
                "get_paper_details": "GET /api/papers/{paper_id}",
                "delete_paper": "DELETE /api/papers/{paper_id}",
                "paper_statistics": "GET /api/papers/{paper_id}/stats"
            },
            "analytics": {
                "query_history": "GET /api/queries/history",
                "popular_queries": "GET /api/analytics/popular",
                "paper_usage": "GET /api/analytics/papers",
                "performance_metrics": "GET /api/analytics/performance"
            },
            "system": {
                "health_check": "GET /api/health",
                "documentation": "GET /docs",
                "openapi_schema": "GET /openapi.json"
            }
        },
        "features": [
            "Semantic search with vector embeddings",
            "LLM-powered answer generation",
            "Automatic citation extraction and storage",
            "Query result caching",
            "Batch query processing",
            "User rating system",
            "Comprehensive analytics and performance tracking"
        ]
    }
