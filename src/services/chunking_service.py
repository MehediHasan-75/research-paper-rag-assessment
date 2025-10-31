from typing import List, Dict, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import re
import logging
import os
from src.config import settings


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with enhanced metadata"""
    text: str
    section: str
    page_number: int
    chunk_index: int
    metadata: Dict = field(default_factory=dict)
    
    paper_name: Optional[str] = None  # From PDF filename
    section_id: Optional[str] = None  # e.g., "2.1"
    section_level: int = 0  # 0=main, 1=sub, 2=subsub


class IntelligentChunker:
    """
    Production-grade intelligent chunking for research papers with paper_name tracking
    """
    
    def __init__(self, 
                 model_name: str = settings.EMBEDDING_MODEL,
                 chunk_size: int = settings.CHUNK_SIZE,
                 overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if os.getenv('HF_HUB_OFFLINE') == '1' or os.getenv('TRANSFORMERS_OFFLINE') == '1':
            self.tokenizer = None
            logger.info("Offline mode enabled; using word-based token estimation.")
        else:
            try:
                hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
                try:
                    # Prefer full repo id first
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                except TypeError:
                    # Older transformers versions use use_auth_token
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
                logger.info(f"✅ Loaded tokenizer: {model_name}")
            except Exception as e:
                # Fallback to normalized name if the full repo id isn't a tokenizer repo
                try:
                    normalized_name = model_name.replace('sentence-transformers/', '')
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(normalized_name, token=hf_token)
                    except TypeError:
                        self.tokenizer = AutoTokenizer.from_pretrained(normalized_name, use_auth_token=hf_token)
                    logger.info(f"✅ Loaded tokenizer (fallback): {normalized_name}")
                except Exception:
                    logger.warning(f"Failed to load tokenizer: {e}. Falling back to word estimation.")
                    self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """Accurate token estimation"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            return int(len(text.split()) * 1.3)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving academic formatting"""
        sentences = re.split(
            r'(?<!\d)\.(?!\d)(?!\s*al\.)(?!\s*Fig\.)(?!\s*Tab\.)(?!\s*Eq\.)'
            r'(?!\s*Dr\.)(?!\s*Prof\.)(?!\s*Inc\.)(?!\s*Ltd\.)(?!\s*etc\.)\s+',
            text
        )
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_section(self, 
                     section_text: str, 
                     section_name: str, 
                     page_start: int, 
                     page_end: int,
                     paper_name: Optional[str] = None,  
                     section_id: Optional[str] = None,  
                     section_level: int = 0  
                     ) -> List[Chunk]:
        """
        Create optimally-sized chunks from a section with paper_name tracking
        
        Args:
            section_text: Text to chunk
            section_name: Section name
            page_start: Starting page
            page_end: Ending page
            paper_name: ✅ NEW: PDF filename (e.g., "lwe_optimization_2024")
            section_id: ✅ NEW: Section numbering (e.g., "2.1")
            section_level: ✅ NEW: Hierarchy level (0=main, 1=sub)
        """
        sentences = self.split_into_sentences(section_text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if sentence_tokens > self.chunk_size:
                logger.warning(f"Long sentence ({sentence_tokens} tokens) in {section_name}")
                if self.tokenizer:
                    tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                    for i in range(0, len(tokens), self.chunk_size - self.overlap):
                        chunk_tokens = tokens[i:i + self.chunk_size]
                        chunk_text = self.tokenizer.decode(chunk_tokens)
                        chunks.append(Chunk(
                            text=chunk_text,
                            section=section_name,
                            page_number=page_start,
                            chunk_index=chunk_index,
                            metadata={
                                'token_count': len(chunk_tokens),
                                'is_partial_sentence': True,
                                'warning': 'sentence_too_long'
                            },
                            paper_name=paper_name,  
                            section_id=section_id,  
                            section_level=section_level  
                        ))
                        chunk_index += 1
                else:
                    chunks.append(Chunk(
                        text=sentence,
                        section=section_name,
                        page_number=page_start,
                        chunk_index=chunk_index,
                        metadata={
                            'token_count': sentence_tokens,
                            'is_partial_sentence': True,
                            'warning': 'long_sentence_no_tokenizer'
                        },
                        paper_name=paper_name,  
                        section_id=section_id,  
                        section_level=section_level  
                    ))
                    chunk_index += 1
                continue
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    section=section_name,
                    page_number=page_start,
                    chunk_index=chunk_index,
                    metadata={
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk),
                        'is_complete': True
                    },
                    paper_name=paper_name,  
                    section_id=section_id,  
                    section_level=section_level  
                ))
                
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.estimate_tokens(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                section=section_name,
                page_number=page_start,
                chunk_index=chunk_index,
                metadata={
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk),
                    'is_final': True
                },
                paper_name=paper_name,  
                section_id=section_id,  
                section_level=section_level 
            ))
        
        avg_tokens = sum(c.metadata['token_count'] for c in chunks) / len(chunks) if chunks else 0
        logger.info(
            f"Created {len(chunks)} chunks from '{section_name}' "
            f"(avg {avg_tokens:.0f} tokens/chunk) | "  
            f"paper: {paper_name} | section_id: {section_id}"  
        )
        
        return chunks


# ✅ Global instance for easy import
intelligent_chunker = IntelligentChunker()
