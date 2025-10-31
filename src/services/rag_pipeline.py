import ollama
from typing import List, Dict, Optional
import time
from src.models.database import Paper


class RAGPipeline:
    def __init__(self, qdrant_service, embedding_service, db_session, 
                 model: str = "deepseek-r1:8b"):
        self.qdrant = qdrant_service
        self.embedder = embedding_service
        self.db = db_session
        self.model = model
    
    def generate_answer(self, question: str, top_k: int = 5, 
                       paper_ids: Optional[List[int]] = None) -> Dict:
        """Main RAG pipeline with paper_name tracking"""
        start_time = time.time()
        
        # 1. Encode query
        query_vector = self.embedder.encode_single(question)
        
        # 2. Retrieve relevant chunks
        search_results = self.qdrant.search(
            query_vector=query_vector.tolist(),
            top_k=top_k,
            paper_ids=paper_ids
        )
        
        if not search_results:
            return {
                'answer': "No relevant information found in the papers.",
                'citations': [],
                'sources_used': [],
                'confidence': 0.0,
                'response_time': time.time() - start_time
            }
        
        # 3. Build context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            paper = self.db.query(Paper).filter(Paper.id == result['paper_id']).first()
            # ✅ INCLUDE PAPER_NAME IN CONTEXT
            context_parts.append(
                f"[Source {i}] From '{paper.title}' (paper_name: {paper.paper_name}), {result['section']} (Page {result['page']}):\n"
                f"{result['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # 4. Generate answer with LLM
        prompt = f"""You are a research assistant helping analyze academic papers.


Context from research papers:
{context}


Question: {question}


Instructions:
- Provide a clear, accurate answer based ONLY on the context provided
- Cite specific sources using [Source N] notation
- If the context doesn't fully answer the question, acknowledge what's missing
- Be concise but comprehensive


Answer:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        answer = response['message']['content']
        
        # 5. Build citations with paper_name
        citations = []
        for result in search_results:
            paper = self.db.query(Paper).filter(Paper.id == result['paper_id']).first()
            citations.append({
                'paper_title': paper.title,
                'paper_name': paper.paper_name,  # ✅ ADD THIS
                'section': result['section'],
                'page': result['page'],
                'relevance_score': round(result['score'], 3)
            })
        
        # 6. Calculate confidence (average of top 3 scores)
        confidence = sum(r['score'] for r in search_results[:3]) / min(3, len(search_results))
        
        response_time = time.time() - start_time
        
        return {
            'answer': answer,
            'citations': citations,
            'sources_used': list(set(c['paper_title'] for c in citations)),
            'confidence': round(confidence, 2),
            'response_time': round(response_time, 2)
        }
