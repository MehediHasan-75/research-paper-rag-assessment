from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    """Qdrant vector database service"""
    
    COLLECTION_NAME = settings.QDRANT_COLLECTION
    
    def __init__(self, host: str = settings.QDRANT_HOST, port: int = settings.QDRANT_PORT, dimension: int = settings.EMBEDDING_DIMENSION):
        try:
            self.client = QdrantClient(host=host, port=port)
            self.dimension = dimension
            self.ensure_collection()
            logger.info(f"✅ Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.COLLECTION_NAME for c in collections):
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created Qdrant collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
    
    def upsert_chunks(self, 
                     chunks: List[Dict], 
                     embeddings: List, 
                     paper_id: int) -> List[str]:
        """Store chunk vectors in Qdrant"""
        try:
            points = []
            vector_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                
                # ✅ Handle both numpy arrays and lists
                if hasattr(embedding, 'tolist'):
                    vector = embedding.tolist()
                else:
                    if isinstance(embedding, list) and len(embedding) > 0:
                        if isinstance(embedding[0], list):
                            vector = embedding[0]
                        else:
                            vector = embedding
                    else:
                        vector = embedding
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'chunk_id': chunk['id'],
                        'paper_id': paper_id,
                        'text': chunk['text'],
                        'section': chunk['section'],
                        'page': chunk['page_number']
                    }
                ))
                vector_ids.append(point_id)
            
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
            logger.info(f"Upserted {len(points)} vectors for paper {paper_id}")
            return vector_ids
        
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise
    
    def search(self, 
              query_vector: List[float], 
              top_k: int = 5,
              paper_ids: Optional[List[int]] = None) -> List[Dict]:
        """Search similar vectors"""
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )
            
            return [
                {
                    'chunk_id': r.payload['chunk_id'],
                    'paper_id': r.payload['paper_id'],
                    'text': r.payload['text'],
                    'section': r.payload['section'],
                    'page': r.payload['page'],
                    'score': r.score
                }
                for r in results
            ]
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_by_paper(self, paper_id: int):
        """Delete all vectors for a paper"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    )]
                )
            )
            logger.info(f"Deleted vectors for paper {paper_id}")
        except Exception as e:
            logger.error(f"Delete failed: {e}")


# ✅ Global instance for easy import (uses env via settings)
qdrant_service = QdrantService()
