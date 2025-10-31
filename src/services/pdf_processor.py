import pdfplumber
import re
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
from functools import lru_cache
import hashlib


logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of a document"""
    name: str
    text: str
    page_start: int
    page_end: int
    level: int = 0  # 0 for main sections, 1 for subsections, etc.
    section_id: Optional[str] = None  # e.g., "2.1", "3.2.1"


@dataclass
class Citation:
    """Represents an extracted citation"""
    text: str
    line_number: int
    page: int


@dataclass
class ProcessedDocument:
    """Complete processed document with metadata"""
    title: str
    paper_name: str  
    authors: List[str]
    year: int
    abstract: str
    sections: List[DocumentSection]
    total_pages: int
    format_type: str  # "standard", "commentary", "report"
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class EnhancedTextCleaner:
    """
    Cleans extracted text for better quality
    Handles: special characters, hyphenation, ligatures, multiple spaces
    """
    
    # Common ligatures
    LIGATURES = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
    }
    
    # Special dashes/hyphens
    DASH_VARIANTS = {
        '–': '-',  # en-dash
        '—': '-',  # em-dash
        '−': '-',  # minus
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Multi-stage text cleaning"""
        if not text:
            return ""
        
        # Stage 1: Replace ligatures
        for ligature, replacement in EnhancedTextCleaner.LIGATURES.items():
            text = text.replace(ligature, replacement)
        
        # Stage 2: Normalize dashes
        for dash, replacement in EnhancedTextCleaner.DASH_VARIANTS.items():
            text = text.replace(dash, replacement)
        
        # Stage 3: Fix hyphenation (e.g., "opti-\nmization" → "optimization")
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Stage 4: Normalize whitespace (preserve paragraph breaks)
        lines = text.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(lines)
        
        # Stage 5: Remove excessive blank lines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Stage 6: Fix common OCR errors
        text = re.sub(r'(\d)\s+([ivxlcm])\b', r'\1\2', text)  
        
        return text
    
    @staticmethod
    def preserve_lists(text: str) -> Tuple[str, List[str]]:
        """Extract and preserve bullet points/lists"""
        list_patterns = [
            r'^\s*[•\-\*]\s+(.+)$',
            r'^\s*\d+\.\s+(.+)$',
            r'^\s*[a-z]\)\s+(.+)$',
        ]
        
        lists = []
        for pattern in list_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            lists.extend(matches)
        
        return text, lists


class AuthorExtractor:
    """
    Enhanced author extraction with name parsing
    Handles: multiple formats, affiliations, et al., author lists
    """
    
    @staticmethod
    def extract_authors(text: str, max_authors: int = 10) -> List[str]:
        """
        Extract authors with improved heuristics
        """
        lines = text.split('\n')[:15]  # Check first 15 lines
        
        authors = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip common non-author lines
            if any(skip in stripped.lower() for skip in [
                'abstract', 'introduction', 'http', 'email', '©', 'doi',
                'corresponding', 'affiliation', 'department', 'university'
            ]):
                continue
            
            # Skip if too short or too long
            if len(stripped) < 5 or len(stripped) > 200:
                continue
            
            # Check if line looks like author (multiple capitalized words)
            words = stripped.split()
            capitalized = sum(1 for w in words if w and w[0].isupper())
            
            # Also check for common name patterns
            has_comma = ',' in stripped
            has_and = ' and ' in stripped.lower() or ' & ' in stripped
            
            if (capitalized >= 2 and len(words) >= 2) or has_comma or has_and:
                # Split multiple authors if separated by "and" or "&"
                author_list = re.split(r'\s+and\s+|\s+&\s+', stripped, flags=re.IGNORECASE)
                
                for author in author_list:
                    author = author.strip()
                    # Remove superscript numbers and affiliations
                    author = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰†‡*]+', '', author)
                    author = re.sub(r'\s*\([^)]*\)', '', author)
                    
                    if len(author) > 2 and len(authors) < max_authors:
                        authors.append(author)
        
        return list(dict.fromkeys(authors))[:max_authors]  # Remove duplicates


class SectionHierarchyDetector:
    """
    Detects and maintains section hierarchy (main, sub, subsub)
    """
    
    @staticmethod
    def parse_section_id(header: str) -> Optional[str]:
        """
        Extract section numbering (e.g., "2.1" from "2.1 Related Work")
        Returns: "2.1" or None
        """
        match = re.match(r'^(\d+(?:\.\d+)*)\s+', header)
        return match.group(1) if match else None
    
    @staticmethod
    def get_section_level(section_id: str) -> int:
        """
        Get hierarchy level from section ID
        "1" → 0, "2.1" → 1, "3.2.1" → 2
        """
        if not section_id:
            return 0
        return section_id.count('.') 


class PDFProcessorEnhanced:
    """
    Enhanced production-grade PDF processor for research papers
    
    Improvements:
    - Better metadata extraction with author parsing
    - Section hierarchy detection
    - Text quality metrics
    - Enhanced text cleaning
    - Keywords extraction
    - Citation detection
    - Paper name from PDF filename
    - Extraction quality assessment
    """
    
    # ✅ EXPANDED SECTION PATTERNS
    SECTION_PATTERNS = [
        # Main sections - Abstract
        r'^\s*(?:abstract|summary|executive\s*summary)\s*$',
        
        # Introduction variants
        r'^\s*(?:introduction|overview|background|motivation|preliminaries)\s*$',
        
        # Related Work/Literature Review
        r'^\s*(?:related\s*works?|literature\s*review|state\s*of\s*the\s*art|prior\s*work)\s*$',
        
        # Methodology/Methods/Approach
        r'^\s*(?:methodology|methods?|approach|technical\s*approach|proposal|framework|design)\s*$',
        
        # Results/Experiments/Evaluation
        r'^\s*(?:results?|experiments?|evaluation|findings|empirical\s*study|benchmarks?)\s*$',
        
        # Discussion/Analysis/Implications
        r'^\s*(?:discussion|analysis|implications|findings|observations)\s*$',
        
        # Conclusion/Future Work
        r'^\s*(?:conclusions?|future\s*work|limitations|summary|recommendations)\s*$',
        
        # References/Bibliography
        r'^\s*(?:references|bibliography|works\s*cited|citations)\s*$',
        
        # Appendix/Supplementary
        r'^\s*(?:appendix|supplementary|supplement|appendices|supplemental)\s*$',
        r'^\s*appendix\s+[a-z]\s*:?\s*$',
        
        # Numbered sections (1., 1), 1:, etc.)
        r'^\s*\d+\.?\s+[A-Z][\w\s&-]+$',
        
        # Nested sections (2.1, 3.2.1, etc.)
        r'^\s*\d+\.\d+(?:\.\d+)?\s+[A-Z][\w\s&-]+$',
        
        # Section headers with ALL CAPS
        r'^\s*[A-Z][A-Z\s]{5,}$',
        
        # Keywords, acknowledgments
        r'^\s*(?:keywords?|acknowledgments?|authors?|contributors?)\s*$',
    ]
    
    def __init__(self, enable_caching: bool = True):
        self.section_regex = re.compile(
            '|'.join(self.SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
        self.enable_caching = enable_caching
        self._cache: Dict[str, ProcessedDocument] = {}
        self.text_cleaner = EnhancedTextCleaner()
        self.author_extractor = AuthorExtractor()
        self.hierarchy_detector = SectionHierarchyDetector()
    
    def _get_file_hash(self, pdf_path: str) -> str:
        """Generate hash of file for caching"""
        return hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
    
    def extract_metadata(self, pdf_path: str) -> Tuple[str, List[str], int, List[str]]:
        """
        Extract title, authors, year, keywords from first page
        
        Returns: (title, authors, year, keywords)
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0].extract_text() or ""
                lines = [l.strip() for l in first_page.split('\n') if l.strip()]
                
                if not lines:
                    logger.warning(f"Empty first page in {pdf_path}")
                    return "Unknown", [], 2024, []
                
                # ✅ Title: usually first substantial line
                title = lines[0]
                
                # ✅ Enhanced author extraction
                authors = self.author_extractor.extract_authors(first_page)
                
                # ✅ Year: search entire first page
                year_match = re.search(r'\(?(19|20)\d{2}\)?', first_page)
                year = int(year_match.group(1)) if year_match else 2024
                
                # ✅ Keywords: look for keywords section
                keywords = []
                keywords_match = re.search(
                    r'(?:keywords?|key\s*terms?)\s*:?\s*([^\n]+)',
                    first_page,
                    re.IGNORECASE
                )
                if keywords_match:
                    keywords_text = keywords_match.group(1)
                    keywords = [k.strip() for k in keywords_text.split(',')][:10]
                
                return title, authors, year, keywords
        
        except Exception as e:
            logger.error(f"Metadata extraction failed for {pdf_path}: {e}")
            return "Unknown", [], 2024, []
    
    def extract_sections(self, pdf_path: str) -> Tuple[List[DocumentSection], str]:
        """
        Extract sections with hierarchy detection
        Returns: (sections, format_type)
        """
        sections = []
        current_section = None
        current_text = []
        current_page = 0
        section_count = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    
                    # ✅ Clean extracted text
                    text = self.text_cleaner.clean_text(text)
                    lines = text.split('\n')
                    
                    for line in lines:
                        if self.section_regex.search(line):
                            # Save previous section
                            if current_section and current_text:
                                section_text = '\n'.join(current_text).strip()
                                
                                if len(section_text) > 20:
                                    # ✅ Detect hierarchy level
                                    section_id = self.hierarchy_detector.parse_section_id(current_section)
                                    level = self.hierarchy_detector.get_section_level(section_id)
                                    
                                    sections.append(DocumentSection(
                                        name=current_section,
                                        text=section_text,
                                        page_start=current_page,
                                        page_end=page_num,
                                        level=level,
                                        section_id=section_id
                                    ))
                                    section_count += 1
                            
                            current_section = line.strip()
                            current_text = []
                            current_page = page_num
                        
                        elif current_section:
                            stripped = line.strip()
                            
                            # ✅ Enhanced noise filtering
                            if self._is_noise(stripped):
                                continue
                            
                            current_text.append(stripped)
        
        except Exception as e:
            logger.error(f"Section extraction failed for {pdf_path}: {e}")
        
        # Add final section
        if current_section and current_text:
            section_text = '\n'.join(current_text).strip()
            if len(section_text) > 20:
                section_id = self.hierarchy_detector.parse_section_id(current_section)
                level = self.hierarchy_detector.get_section_level(section_id)
                
                sections.append(DocumentSection(
                    name=current_section,
                    text=section_text,
                    page_start=current_page,
                    page_end=total_pages,
                    level=level,
                    section_id=section_id
                ))
                section_count += 1
        
        # Determine format type
        if section_count >= 5:
            format_type = "standard"
        elif section_count >= 2:
            format_type = "report"
        else:
            format_type = "commentary"
        
        # Fallback for papers with no formal sections
        if not sections:
            logger.warning(f"No standard sections in {pdf_path}. Using fallback.")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = '\n'.join(
                        (page.extract_text() or "") for page in pdf.pages
                    )
                    full_text = self.text_cleaner.clean_text(full_text)
                    sections = [DocumentSection(
                        name="Full Document",
                        text=full_text,
                        page_start=1,
                        page_end=total_pages
                    )]
            except Exception as e:
                logger.error(f"Fallback extraction failed: {e}")
        
        return sections, format_type
    
    def _is_noise(self, line: str) -> str:
        """Filter noise from extracted text"""
        # Empty or too short
        if not line or len(line) < 3:
            return True
        
        # Page numbers
        if re.match(r'^\d+$', line):
            return True
        
        # Just symbols/dashes
        if re.match(r'^[-\s\d.(),|]+$', line):
            return True
        
        # Common footer/header patterns
        if any(footer in line.lower() for footer in [
            'page', 'copyright', 'all rights', 'doi:', 'arxiv', 'proceedings'
        ]):
            return True
        
        return False
    
    def extract_citations(self, sections: List[DocumentSection]) -> List[Citation]:
        """
        Extract potential citations from References section
        """
        citations = []
        
        for section in sections:
            if 'reference' not in section.name.lower():
                continue
            
            lines = section.text.split('\n')
            for i, line in enumerate(lines):
                # Simple citation pattern: starts with [number] or number.
                if re.match(r'^\s*\[\d+\]|\d+\.\s', line.strip()):
                    citations.append(Citation(
                        text=line.strip(),
                        line_number=i,
                        page=section.page_start
                    ))
        
        return citations
    
    def calculate_quality_metrics(self, doc: ProcessedDocument) -> Dict[str, float]:
        """
        Calculate extraction quality metrics
        """
        metrics = {}
        
        # Metadata completeness (0-100)
        completeness = 0
        completeness += 25 if doc.title != "Unknown" else 0
        completeness += 25 if len(doc.authors) > 0 else 0
        completeness += 25 if doc.year != 2024 else 0
        completeness += 25 if doc.abstract else 0
        metrics['metadata_completeness'] = completeness
        
        # Section extraction quality (0-100)
        section_quality = min(100, (len(doc.sections) / 7) * 100)
        metrics['section_extraction_quality'] = section_quality
        
        # Content density (average chars per section)
        if doc.sections:
            avg_section_length = sum(len(s.text) for s in doc.sections) / len(doc.sections)
            metrics['avg_section_length'] = avg_section_length
        
        # Format type confidence
        format_scores = {
            'standard': 95,
            'report': 85,
            'commentary': 70
        }
        metrics['format_confidence'] = format_scores.get(doc.format_type, 50)
        
        # Overall quality score
        metrics['overall_quality'] = (
            completeness * 0.3 +
            section_quality * 0.4 +
            metrics.get('format_confidence', 50) * 0.3
        ) / 100
        
        return metrics
    
    def process_document(self, pdf_path: str) -> Optional[ProcessedDocument]:
        """
        Main processing pipeline
        ✅ NEW: Includes paper_name from PDF filename
        """
        try:
            # Check cache
            file_hash = self._get_file_hash(pdf_path) if self.enable_caching else None
            if file_hash and file_hash in self._cache:
                logger.info(f"Returning cached result for {pdf_path}")
                return self._cache[file_hash]
            
            # ✅ Extract paper name from filename
            paper_name = Path(pdf_path).stem  # Filename without extension
            
            # Extract metadata
            title, authors, year, keywords = self.extract_metadata(pdf_path)
            
            # Extract sections
            sections, format_type = self.extract_sections(pdf_path)
            
            # Extract abstract
            abstract = ""
            for section in sections:
                if 'abstract' in section.name.lower():
                    abstract = section.text[:500]
                    break
            
            # Extract citations
            citations = self.extract_citations(sections)
            
            # Get total pages
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
            
            # Create document
            doc = ProcessedDocument(
                title=title,
                paper_name=paper_name,
                authors=authors,
                year=year,
                abstract=abstract,
                sections=sections,
                total_pages=total_pages,
                format_type=format_type,
                citations=citations,
                keywords=keywords
            )
            
            # Calculate quality metrics
            doc.quality_metrics = self.calculate_quality_metrics(doc)
            
            # Cache result
            if file_hash:
                self._cache[file_hash] = doc
            
            logger.info(
                f"✅ Processed: {paper_name} | {title[:40]}... | "
                f"Sections: {len(sections)} | Pages: {total_pages} | "
                f"Quality: {doc.quality_metrics['overall_quality']:.2%}"
            )
            
            return doc
        
        except Exception as e:
            logger.error(f"Document processing failed for {pdf_path}: {e}")
            return None
    
    def batch_process(self, pdf_paths: List[str]) -> List[ProcessedDocument]:
        """Process multiple documents"""
        results = []
        for pdf_path in pdf_paths:
            doc = self.process_document(pdf_path)
            if doc:
                results.append(doc)
        return results


# ✅ Global instance for easy import
pdf_processor = PDFProcessorEnhanced()