from typing import List
import numpy as np
import logging
import os
from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    SentenceTransformer Embedding Service
    Encodes text to dense vector embeddings for semantic search
    """
    
    def __init__(
        self, 
        model_name: str = None,
        dimension: int = None,
        batch_size: int = None
    ):
        """
        Initialize EmbeddingService
        
        Args:
            model_name: HuggingFace model name (default: from settings)
            dimension: Embedding dimension (default: from settings)
            batch_size: Batch size for encoding (default: from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.model = None
        
        # Set HuggingFace token
        hf_token = settings.HUGGING_FACE_HUB_TOKEN
        if hf_token:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
            logger.info("✅ HuggingFace token configured")
        else:
            logger.warning("⚠️ No HuggingFace token provided - may fail for private models")
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load SentenceTransformer model"""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"📥 Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded successfully (dimension: {self.dimension})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Cannot load embedding model: {str(e)}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding vector
        
        Args:
            text: Input text to encode
            
        Returns:
            np.ndarray: Normalized embedding of shape (dimension,)
        """
        if self.model is None:
            self._load_model()
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.debug(f"✅ Encoded text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Encoding failed: {str(e)}")
            raise RuntimeError(f"Failed to encode text: {str(e)}")
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode batch of texts to embeddings
        
        Args:
            texts: List of text strings to encode
            show_progress: Show progress bar during encoding
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), dimension)
        """
        if self.model is None:
            self._load_model()
        
        if not texts:
            logger.warning("⚠️ Empty text list provided")
            return np.array([])
        
        try:
            logger.info(f"📥 Encoding batch of {len(texts)} texts (batch_size: {self.batch_size})")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            
            logger.info(f"✅ Batch encoding complete. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch encoding failed: {str(e)}")
            raise RuntimeError(f"Failed to encode batch: {str(e)}")
    
    def encode_with_texts(self, texts: List[str]) -> List[dict]:
        """
        Encode texts and return with original text
        
        Args:
            texts: List of text strings
            
        Returns:
            List of dicts with 'text' and 'embedding' keys
        """
        embeddings = self.encode_batch(texts)
        return [
            {
                'text': text,
                'embedding': emb
            }
            for text, emb in zip(texts, embeddings)
        ]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score between -1 and 1
        """
        return float(np.dot(embedding1, embedding2))
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name


# Global instance
embedding_service = EmbeddingService()
