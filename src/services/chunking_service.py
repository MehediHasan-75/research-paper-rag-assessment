from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer
import re
import logging
import os
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    section: str
    page_number: int
    chunk_index: int
    metadata: Dict

class IntelligentChunker:
    """
    Production-grade intelligent chunking for research papers
    
    Strategy:
    - 450 tokens per chunk (safe buffer below 512 limit)
    - 50 token overlap (context preservation)
    - Sentence-aware (semantic boundaries)
    - Citation-aware (doesn't break [1.2.3] format)
    - Real tokenizer for accuracy (not word approximation)
    - Handles long sentences (>512 tokens)
    
    Why optimal:
    - Sentence-transformers work best at 256-512 tokens
    - LLM context size safe
    - Preserves semantic meaning
    - Academic paper friendly
    """
    
    def __init__(self, 
                 model_name: str = settings.EMBEDDING_MODEL,
                 chunk_size: int = settings.CHUNK_SIZE,
                 overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # ✅ Respect offline mode: skip remote tokenizer load
        if os.getenv('HF_HUB_OFFLINE') == '1' or os.getenv('TRANSFORMERS_OFFLINE') == '1':
            self.tokenizer = None
            logger.info("Offline mode enabled; using word-based token estimation.")
        else:
            # ✅ Use REAL tokenizer for accurate token counting
            try:
                normalized_name = model_name.replace('sentence-transformers/', '')
                self.tokenizer = AutoTokenizer.from_pretrained(normalized_name)
                logger.info(f"✅ Loaded tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Falling back to word estimation.")
                self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Accurate token estimation
        Uses real tokenizer if available, fallback to 1.3x multiplier
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback: ~1.3 tokens per word (conservative)
            return int(len(text.split()) * 1.3)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving academic formatting:
        - Citations [1.2.3]
        - Numbers 2.5, 3.14
        - Abbreviations et al., Fig., Tab., Eq.
        - Common patterns Dr., Prof., Inc.
        """
        # ✅ Enhanced regex for academic papers
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
                     page_end: int) -> List[Chunk]:
        """
        Create optimally-sized chunks from a section
        
        Process:
        1. Split into sentences (semantic boundaries)
        2. Group sentences until chunk_size reached
        3. Add overlap for context preservation
        4. Handle edge cases (long sentences)
        5. Track metadata
        """
        sentences = self.split_into_sentences(section_text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # ✅ EDGE CASE: Single sentence exceeds limit
            # ✅ CORRECT (fixed)
            if sentence_tokens > self.chunk_size:
                logger.warning(f"Long sentence...")
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
                            }
                        ))
                        chunk_index += 1
                else:
                    # If no tokenizer, just add the whole sentence
                    chunks.append(Chunk(
                        text=sentence,
                        section=section_name,
                        page_number=page_start,
                        chunk_index=chunk_index,
                        metadata={
                            'token_count': sentence_tokens,
                            'is_partial_sentence': True,
                            'warning': 'long_sentence_no_tokenizer'
                        }
                    ))
                    chunk_index += 1
                continue  # ✅ NOW OK: we already added chunks above

            
            # ✅ NORMAL CHUNKING: Group sentences
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
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
                    }
                ))
                
                # ✅ Start new chunk with SMART overlap
                # Calculate overlap by sentences, not fixed count
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
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # ✅ Add remaining/final chunk
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
                }
            ))
        
        # Log statistics
        avg_tokens = sum(c.metadata['token_count'] for c in chunks) / len(chunks) if chunks else 0
        logger.info(
            f"Created {len(chunks)} chunks from '{section_name}' "
            f"(avg {avg_tokens:.0f} tokens/chunk)"
        )
        
        return chunks


# ✅ Global instance for easy import
intelligent_chunker = IntelligentChunker()