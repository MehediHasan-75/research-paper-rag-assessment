from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path  # ✅ ADD THIS


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
    paper_title_to_id_map: dict,
    paper_name_to_id_map: dict = None
):
    """Helper to save citations from RAG result to database"""
    
    # ✅ TRACK WHICH PAPERS WE'VE ALREADY CITED
    cited_papers = set()
    
    for citation_data in rag_result.get('citations', []):
        paper_title = citation_data.get('paper_title')
        
        paper_id = None
        if paper_name_to_id_map and citation_data.get('paper_name'):
            paper_id = paper_name_to_id_map.get(citation_data.get('paper_name'))
        
        if not paper_id:
            paper_id = paper_title_to_id_map.get(paper_title)
        
        if not paper_id:
            logger.warning(f"⚠️ Paper not found for citation: {paper_title}")
            continue
        
        # ✅ SKIP IF WE ALREADY CITED THIS PAPER FOR THIS QUERY
        if (query_id, paper_id) in cited_papers:
            logger.info(f"⊘ Citation already exists for query {query_id}, paper {paper_id}")
            continue
        
        try:
            # ✅ Use get_or_create pattern to avoid duplicates
            existing = db.query(Citation).filter(
                Citation.query_id == query_id,
                Citation.paper_id == paper_id
            ).first()
            
            if existing:
                # Update relevance score with highest value
                existing.relevance_score = max(
                    existing.relevance_score,
                    citation_data.get('relevance_score', 0.0)
                )
                logger.info(f"⊘ Updated existing citation: query {query_id}, paper {paper_id}")
            else:
                # Create new citation
                citation = Citation(
                    query_id=query_id,
                    paper_id=paper_id,
                    chunk_id=None,
                    relevance_score=citation_data.get('relevance_score', 0.0)
                )
                db.add(citation)
                cited_papers.add((query_id, paper_id))
        
        except Exception as e:
            logger.error(f"❌ Failed to save citation: {e}")
    
    db.commit()
    logger.info(f"✅ Saved {len(cited_papers)} unique citations")



# ========== DOCUMENT INGESTION SYSTEM ==========


# src/api/routes.py - FIXED UPLOAD ENDPOINT

@router.post("/papers/upload")
async def upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process PDF with paper_name extraction from filename"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    if file.size and file.size > settings.MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    logger.info(f"📄 Uploading: {file.filename}")
    
    # ✅ EXTRACT PAPER_NAME FROM FILENAME (without extension)
    paper_name = Path(file.filename).stem
    logger.info(f"📝 Paper name extracted: {paper_name}")
    
    # ✅ CHECK FOR DUPLICATES BY BOTH paper_name AND filename
    existing_by_name = db.query(Paper).filter(Paper.paper_name == paper_name).first()
    existing_by_filename = db.query(Paper).filter(Paper.filename == file.filename).first()
    
    if existing_by_name or existing_by_filename:
        existing = existing_by_name or existing_by_filename
        logger.warning(f"⚠️ Paper already exists: {paper_name} (ID={existing.id})")
        return {
            "paper_id": existing.id,
            "paper_name": existing.paper_name,
            "title": existing.title,
            "status": "already_exists",
            "message": "This paper was already uploaded",
            "uploaded": existing.upload_date.isoformat()
        }
    
    file_path = os.path.join(settings.UPLOAD_DIR, str(uuid.uuid4()) + ".pdf")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"❌ File save failed: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # Process PDF
    try:
        processed_doc = pdf_processor.process_document(file_path)
        if not processed_doc:
            os.remove(file_path)
            logger.error("PDF processing returned None")
            raise HTTPException(status_code=500, detail="PDF processing failed")
    except Exception as e:
        os.remove(file_path)
        logger.error(f"❌ PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    try:
        # ✅ SAVE PAPER WITH PAPER_NAME AND ALL METADATA
 # ...
        paper = Paper(
            paper_name=paper_name,
            title=processed_doc.title or "Unknown",
            authors=processed_doc.authors or [],
            year=processed_doc.year,
            filename=file.filename,
            file_path=file_path,
            total_pages=processed_doc.total_pages or 0,
            abstract=processed_doc.abstract or "",
            sections=[
                {
                    "name": section.name,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "section_id": getattr(section, "section_id", None),
                    "level": getattr(section, "level", 0),
                }
                for section in processed_doc.sections
            ] if processed_doc.sections else [],
            quality_score=processed_doc.quality_metrics.get('overall_quality', 0),
            keywords=processed_doc.keywords or [],
            format_type=processed_doc.format_type or "standard",
            processed=False
        )

        db.add(paper)
        db.commit()
        db.refresh(paper)
        logger.info(f"✅ Paper saved: ID={paper.id}, paper_name={paper_name}, quality={paper.quality_score:.2f}")
    except Exception as e:
        db.rollback()
        os.remove(file_path)
        logger.error(f"❌ Failed to save paper: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save paper: {str(e)}")
    
    # Step 1: Create chunks with BATCH COMMITS
    chunk_count = 0
    try:
        for section in processed_doc.sections:
            chunks = intelligent_chunker.chunk_section(
                section_text=section.text,
                section_name=section.name,
                page_start=section.page_start,
                page_end=section.page_end,
                paper_name=paper_name,  # ✅ ADD THIS
                section_id=section.section_id,  # ✅ ADD THIS
                section_level=section.level  # ✅ ADD THIS
            )
            
            for chunk in chunks:
                db_chunk = Chunk(
                    paper_id=paper.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    section_id=chunk.section_id,  # ✅ ADD THIS
                    section_level=chunk.section_level  # ✅ ADD THIS
                )
                db.add(db_chunk)
                chunk_count += 1
                
                # BATCH COMMIT: Every 10 chunks
                if chunk_count % 10 == 0:
                    db.commit()
                    logger.info(f"✅ Committed {chunk_count} chunks...")
        
        db.commit()
        logger.info(f"✅ All {chunk_count} chunks committed")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Chunk creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk creation failed: {str(e)}")
    
    # Step 2: Get chunk IDs from database
    try:
        db_chunks = db.query(Chunk).filter(Chunk.paper_id == paper.id).all()
        logger.info(f"✅ Retrieved {len(db_chunks)} chunks from database")
        
        if not db_chunks:
            logger.warning(f"⚠️ No chunks retrieved for paper {paper_name}")
        
        chunk_dicts = [
            {
                'id': c.id,
                'text': c.text,
                'section': c.section,
                'page_number': c.page_number,
                'paper_name': paper_name,  # ✅ ADD THIS
                'section_id': c.section_id,  # ✅ ADD THIS
                'section_level': c.section_level  # ✅ ADD THIS
            }
            for c in db_chunks
        ]
    except Exception as e:
        logger.error(f"❌ Failed to retrieve chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")
    
    # Step 3: Generate embeddings
    try:
        embedding_texts = [c['text'] for c in chunk_dicts]
        embeddings = embedding_service.encode_batch(embedding_texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    # Step 4: Store in Qdrant
    try:
        vector_ids = qdrant_service.upsert_chunks(chunk_dicts, embeddings, paper.id)
        logger.info(f"✅ Stored {len(vector_ids)} vectors in Qdrant")
    except Exception as e:
        logger.error(f"❌ Qdrant storage failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")
    
    # Step 5: Update vector IDs
    try:
        for db_chunk, vector_id in zip(db_chunks, vector_ids):
            db_chunk.vector_id = vector_id
            db_chunk.embedding_generated = True
        
        db.commit()
        logger.info(f"✅ Updated {len(db_chunks)} vector IDs")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to update vector IDs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update vector IDs: {str(e)}")
    
    # Step 6: Mark as processed
    try:
        paper.processed = True
        paper.chunk_count = len(db_chunks)
        db.commit()
        logger.info(f"✅ Paper marked as processed")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to mark paper as processed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark paper as processed: {str(e)}")
    
    logger.info(f"✅ Upload complete: {len(db_chunks)} chunks, quality={paper.quality_score:.2f}")
    
    return {
        "paper_id": paper.id,
        "paper_name": paper_name,  # ✅ ADD THIS
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "keywords": paper.keywords,  # ✅ ADD THIS
        "quality_score": paper.quality_score,  # ✅ ADD THIS
        "format_type": paper.format_type,  # ✅ ADD THIS
        "sections_extracted": len(processed_doc.sections),
        "chunks_created": len(db_chunks),
        "pages": paper.total_pages,
        "upload_status": "success",
        "message": f"Successfully uploaded and processed {paper_name}"
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
    """🚀 Full RAG-Powered Query Endpoint with paper_name in sources"""
    logger.info(f"🔍 RAG Query: {question[:50]}...")
    
    if use_cache:
        cached = cache_service.get_query_cache(question, paper_ids)
        if cached:
            logger.info("✅ Cache hit")
            cached['cached'] = True
            return cached
    
    try:
        rag_pipeline = get_rag_pipeline(db)
        rag_result = rag_pipeline.generate_answer(question=question, top_k=top_k, paper_ids=paper_ids)
        
        # ✅ CREATE PAPER_NAME MAPPING
        paper_title_to_id_map = {}
        paper_title_to_name_map = {}
        
        for citation_data in rag_result.get('citations', []):
            paper_title = citation_data.get('paper_title')
            paper = db.query(Paper).filter(Paper.title == paper_title).first()
            if paper:
                paper_title_to_id_map[paper_title] = paper.id
                paper_title_to_name_map[paper_title] = paper.paper_name
        
        query_id = _save_query_history(
            db=db, query_text=question, top_k=top_k,
            response_time=rag_result['response_time'],
            confidence=rag_result['confidence'],
            answer=rag_result['answer'],
            paper_filter=paper_ids
        )
        
        _save_citations(db, query_id, rag_result, paper_title_to_id_map)
        
        # ✅ UPDATE sources_used to use paper_name
        rag_result['sources_used'] = list(set(
            paper_title_to_name_map.get(c.get('paper_title'), c.get('paper_title'))
            for c in rag_result.get('citations', [])
        ))
        
        # ✅ UPDATE citations to use paper_name
        for citation in rag_result.get('citations', []):
            paper_title = citation.get('paper_title')
            paper_name = paper_title_to_name_map.get(paper_title, paper_title)
            citation['paper_name'] = paper_name
        
        if use_cache:
            cache_service.set_query_cache(question, rag_result, paper_ids)
        
        rag_result['cached'] = False
        logger.info(f"✅ Query answered in {rag_result['response_time']}s. Sources: {rag_result['sources_used']}")
        return rag_result
        
    except Exception as e:
        logger.error(f"❌ RAG query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


 


@router.post("/query/batch")
async def rag_query_batch(
    questions: List[str] = Body(..., min_items=1, max_items=10),
    top_k: int = Body(5, ge=1, le=20),
    paper_ids: Optional[List[int]] = Body(None),
    db: Session = Depends(get_db)
):
    """⚡ Batch RAG Query Processing with paper_name in sources"""
    logger.info(f"⚡ Processing batch of {len(questions)} queries")
    
    results = []
    rag_pipeline = get_rag_pipeline(db)
    
    # ✅ CREATE PAPER_NAME MAPPING ONCE
    paper_title_to_id_map = {}
    paper_title_to_name_map = {}
    all_papers = db.query(Paper).all()
    for paper in all_papers:
        paper_title_to_id_map[paper.title] = paper.id
        paper_title_to_name_map[paper.title] = paper.paper_name
    
    for idx, question in enumerate(questions, 1):
        try:
            logger.info(f"Processing query {idx}/{len(questions)}")
            result = rag_pipeline.generate_answer(question=question, top_k=top_k, paper_ids=paper_ids)
            
            query_id = _save_query_history(
                db=db, query_text=question, top_k=top_k,
                response_time=result['response_time'],
                confidence=result['confidence'],
                answer=result['answer'],
                paper_filter=paper_ids
            )
            
            _save_citations(db, query_id, result, paper_title_to_id_map)
            
            # ✅ UPDATE sources_used
            result['sources_used'] = list(set(
                paper_title_to_name_map.get(c.get('paper_title'), c.get('paper_title'))
                for c in result.get('citations', [])
            ))
            
            # ✅ UPDATE citations
            for citation in result.get('citations', []):
                paper_title = citation.get('paper_title')
                paper_name = paper_title_to_name_map.get(paper_title, paper_title)
                citation['paper_name'] = paper_name
            
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




@router.get("/papers")
async def list_papers(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    min_quality: Optional[float] = Query(None, ge=0, le=1, description="Filter by minimum quality"),  # ✅ ADD
    db: Session = Depends(get_db)
):
    """List all uploaded papers with pagination and quality filtering"""
    query = db.query(Paper)
    
    # ✅ ADD QUALITY FILTERING
    if min_quality is not None:
        query = query.filter(Paper.quality_score >= min_quality)
    
    papers = query.offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "papers": [
            {
                "id": p.id,
                "paper_name": p.paper_name,  # ✅ ADD THIS
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "pages": p.total_pages,
                "chunks": p.chunk_count,
                "quality_score": p.quality_score,  # ✅ ADD THIS
                "keywords": p.keywords,  # ✅ ADD THIS
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
        "paper_name": paper.paper_name,  # ✅ ADD THIS
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "keywords": paper.keywords,  # ✅ ADD THIS
        "quality_score": paper.quality_score,  # ✅ ADD THIS
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
    
    try:
        qdrant_service.delete_by_paper(paper_id)
    except Exception as e:
        logger.warning(f"Qdrant deletion failed: {e}")
    
    db.delete(paper)
    db.commit()
    
    logger.info(f"✅ Paper deleted: {paper.title}")
    
    return {
        "id": paper_id,
        "paper_name": paper.paper_name,  # ✅ ADD THIS
        "status": "deleted",
        "title": paper.title
    }



@router.get("/papers/{paper_id}/stats")
async def paper_stats(paper_id: int, db: Session = Depends(get_db)):
    """View paper statistics"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    citations_count = db.query(Citation).filter(Citation.paper_id == paper_id).count()
    
    from sqlalchemy import func
    query_count = db.query(func.count(QueryHistory.id)).filter(
        text(f"paper_filter::text LIKE '%{paper_id}%'")
    ).scalar() or 0
    
    return {
        "paper_id": paper_id,
        "paper_name": paper.paper_name,  # ✅ ADD THIS
        "title": paper.title,
        "quality_score": paper.quality_score,  # ✅ ADD THIS
        "keywords": paper.keywords,  # ✅ ADD THIS
        "chunks_created": paper.chunk_count,
        "times_cited": citations_count,
        "times_queried": query_count,
        "pages": paper.total_pages,
        "uploaded": paper.upload_date.isoformat()
    }



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
    """Paper usage statistics with paper_name"""
    from sqlalchemy import func
    
    stats = db.query(
        Paper.id,
        Paper.paper_name,  # ✅ ADD THIS
        Paper.title,
        Paper.quality_score,  # ✅ ADD THIS
        func.count(Citation.id).label('citations')
    ).outerjoin(Citation, Paper.id == Citation.paper_id).group_by(
        Paper.id
    ).all()
    
    return {
        "paper_stats": [
            {
                "id": s[0],
                "paper_name": s[1],  # ✅ ADD THIS
                "title": s[2],
                "quality_score": s[3],  # ✅ ADD THIS
                "times_cited": s[4] or 0
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
            "avg_user_rating": None,
            "avg_extraction_quality": None  # ✅ ADD THIS
        }
    
    avg_time = sum(q.response_time or 0 for q in queries) / len(queries)
    avg_conf = sum(q.confidence or 0 for q in queries) / len(queries)
    
    rated_queries = [q for q in queries if q.user_rating is not None]
    avg_rating = sum(q.user_rating for q in rated_queries) / len(rated_queries) if rated_queries else None
    
    # ✅ ADD EXTRACTION QUALITY
    papers = db.query(Paper).all()
    avg_extraction_quality = (
        sum(p.quality_score for p in papers) / len(papers)
        if papers else None
    )
    
    return {
        "total_queries": len(queries),
        "total_papers": db.query(Paper).count(),
        "total_chunks": db.query(Chunk).count(),
        "avg_response_time_ms": round(avg_time * 1000, 2),
        "avg_confidence": round(avg_conf, 3),
        "avg_user_rating": round(avg_rating, 2) if avg_rating else None,
        "rated_responses_count": len(rated_queries),
        "avg_extraction_quality": round(avg_extraction_quality, 3) if avg_extraction_quality else None  # ✅ ADD THIS
    }



@router.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "pdf_processor": "enhanced",  # ✅ UPDATED
            "chunking": "enhanced",  # ✅ UPDATED
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
        "version": "2.1",  # ✅ UPDATED
        "features_enhanced": [  # ✅ ADD THIS
            "Paper name extraction from PDF filenames",
            "Quality score tracking for extraction accuracy",
            "Automatic keywords and citation extraction",
            "Section hierarchy detection (2.1, 3.2.1, etc.)",
            "Format type classification (standard/report/commentary)"
        ],
        "endpoints": {
            "document_ingestion": {
                "upload_paper": "POST /api/papers/upload"
            },
            "rag_queries": {
                "basic_rag_query": "POST /api/query (with query params)",
                
                "batch_rag_query": "POST /api/query/batch"
            },
            "paper_management": {
                "list_papers": "GET /api/papers (with quality filter)",  # ✅ UPDATED
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
            "Comprehensive analytics and performance tracking",
            "✅ Paper name based source tracking",
            "✅ Extraction quality metrics",
            "✅ Section hierarchy support"
        ]
    }
