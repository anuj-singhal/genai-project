"""
RAG Agent for intelligent retrieval-augmented generation
Orchestrates the RAG pipeline with agentic capabilities
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class RAGResponse:
    """RAG response with metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    chunks_used: List[Any]
    confidence: float
    mode: str  # "rag", "direct", or "hybrid"

class RAGAgent:
    """Intelligent RAG agent for document-based Q&A"""
    
    def __init__(self, vector_store, llm_client):
        """
        Initialize RAG agent
        
        Args:
            vector_store: VectorStore instance
            llm_client: OpenAI client instance
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.last_retrieved_chunks = []
    
    def should_use_rag(self, query: str, has_documents: bool) -> Tuple[bool, str]:
        """
        Determine if RAG should be used for the query
        
        Args:
            query: User query
            has_documents: Whether documents are available
            
        Returns:
            Tuple of (should_use_rag, reasoning)
        """
        if not has_documents:
            return False, "No documents available"
        
        # Keywords that suggest document-specific queries
        doc_keywords = [
            "document", "file", "text", "pdf", "according to",
            "based on", "in the", "what does", "find", "search",
            "locate", "where", "mention", "say about", "reference"
        ]
        
        # Check for document-specific keywords
        query_lower = query.lower()
        has_doc_reference = any(keyword in query_lower for keyword in doc_keywords)
        
        # Check if query is asking for specific information
        is_specific_query = any(q in query_lower for q in ["what", "where", "when", "who", "how many", "which"])
        
        if has_doc_reference:
            return True, "Query references documents"
        elif is_specific_query:
            return True, "Query seeks specific information"
        else:
            # For general queries, optionally use RAG
            return True, "Using RAG for enhanced context"
    
    def generate_search_query(self, user_query: str, context: List[str] = None) -> str:
        """
        Generate optimized search query from user input
        
        Args:
            user_query: Original user query
            context: Previous conversation context
            
        Returns:
            Optimized search query
        """
        # For now, use the query as-is
        # In future, could use LLM to reformulate
        return user_query
    
    def rank_chunks(self, chunks: List[Any], query: str) -> List[Any]:
        """
        Re-rank chunks based on relevance to query
        
        Args:
            chunks: Retrieved chunks
            query: User query
            
        Returns:
            Re-ranked chunks
        """
        # Simple re-ranking based on keyword overlap
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            # Combine similarity score with keyword overlap
            combined_score = chunk.similarity_score * 0.7 + (overlap / len(query_words)) * 0.3
            chunk.combined_score = combined_score
            scored_chunks.append(chunk)
        
        # Sort by combined score
        return sorted(scored_chunks, key=lambda x: x.combined_score, reverse=True)
    
    def build_context(self, chunks: List[Any], max_tokens: int = 2000) -> str:
        """
        Build context from retrieved chunks
        
        Args:
            chunks: Retrieved and ranked chunks
            max_tokens: Maximum tokens for context
            
        Returns:
            Combined context string
        """
        context_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(chunks):
            # Approximate token count
            chunk_tokens = len(chunk.content) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            # Format chunk with source
            context_part = f"[Source: {chunk.source}, Score: {chunk.similarity_score:.2f}]\n{chunk.content}"
            context_parts.append(context_part)
            current_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_rag_response(self,
                             query: str,
                             context: str,
                             system_prompt: str,
                             conversation_history: List[Dict] = None,
                             model: str = "gpt-4o-mini",
                             temperature: float = 0.7,
                             max_tokens: int = 1000) -> str:
        """
        Generate response using RAG context
        
        Args:
            query: User query
            context: Retrieved document context
            system_prompt: System prompt
            conversation_history: Previous messages
            model: LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response
        """
        # Build RAG-specific system prompt
        rag_system_prompt = f"""{system_prompt}

You have access to document context that may help answer the user's question. 
Use this context when relevant, but also use your general knowledge when appropriate.
If the context doesn't contain relevant information, you can say so and provide a general answer.
Always cite the source when using information from the documents.

Document Context:
{context}"""
        
        # Build messages
        messages = [
            {"role": "system", "content": rag_system_prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            # Generate response
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating RAG response: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_query(self,
                     query: str,
                     system_prompt: str,
                     initial_knowledge: str = "",
                     conversation_history: List[Dict] = None,
                     model: str = "gpt-4o-mini",
                     temperature: float = 0.7,
                     max_tokens: int = 1000,
                     top_k: int = 5,
                     similarity_threshold: float = 0.3,
                     force_rag: bool = False) -> RAGResponse:
        """
        Process a query with intelligent RAG pipeline
        
        Args:
            query: User query
            system_prompt: System prompt
            initial_knowledge: Initial knowledge context
            conversation_history: Previous conversation
            model: LLM model
            temperature: Generation temperature
            max_tokens: Max tokens for response
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            force_rag: Force use of RAG even if not needed
            
        Returns:
            RAGResponse object
        """
        # Check collection stats
        stats = self.vector_store.get_collection_stats()
        has_documents = stats.get("total_chunks", 0) > 0
        
        # Determine if RAG should be used
        use_rag, reasoning = self.should_use_rag(query, has_documents)
        
        if force_rag and has_documents:
            use_rag = True
            reasoning = "RAG mode forced by user"
        
        if not use_rag:
            # Direct LLM response without RAG
            return RAGResponse(
                answer=self._generate_direct_response(
                    query, system_prompt, initial_knowledge, 
                    conversation_history, model, temperature, max_tokens
                ),
                sources=[],
                chunks_used=[],
                confidence=1.0,
                mode="direct"
            )
        
        # Generate search query
        search_query = self.generate_search_query(query, conversation_history)
        
        # Retrieve chunks
        retrieved_chunks = self.vector_store.search(
            search_query, 
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Store for reference
        self.last_retrieved_chunks = retrieved_chunks
        
        if not retrieved_chunks:
            # No relevant chunks found, fallback to direct response
            return RAGResponse(
                answer=self._generate_direct_response(
                    query, system_prompt, initial_knowledge,
                    conversation_history, model, temperature, max_tokens
                ) + "\n\n*Note: No relevant information found in the uploaded documents.*",
                sources=[],
                chunks_used=[],
                confidence=0.5,
                mode="direct"
            )
        
        # Rank chunks
        ranked_chunks = self.rank_chunks(retrieved_chunks, query)
        
        # Build context
        context = self.build_context(ranked_chunks)
        
        # Add initial knowledge if provided
        if initial_knowledge:
            context = f"{initial_knowledge}\n\n{context}"
        
        # Generate RAG response
        answer = self.generate_rag_response(
            query, context, system_prompt, conversation_history,
            model, temperature, max_tokens
        )
        
        # Extract unique sources
        sources = []
        seen_sources = set()
        for chunk in ranked_chunks[:3]:  # Top 3 sources
            if chunk.source not in seen_sources:
                sources.append({
                    "filename": chunk.source,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "similarity": chunk.similarity_score
                })
                seen_sources.add(chunk.source)
        
        # Calculate confidence based on similarity scores
        avg_similarity = sum(c.similarity_score for c in ranked_chunks[:3]) / min(3, len(ranked_chunks))
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=ranked_chunks[:top_k],
            confidence=avg_similarity,
            mode="rag"
        )
    
    def _generate_direct_response(self,
                                 query: str,
                                 system_prompt: str,
                                 initial_knowledge: str,
                                 conversation_history: List[Dict],
                                 model: str,
                                 temperature: float,
                                 max_tokens: int) -> str:
        """Generate direct LLM response without RAG"""
        # Build system prompt with initial knowledge
        full_system_prompt = system_prompt
        if initial_knowledge:
            full_system_prompt += f"\n\nAdditional Context:\n{initial_knowledge}"
        
        messages = [
            {"role": "system", "content": full_system_prompt}
        ]
        
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_chunk_preview(self, chunk_id: str) -> Optional[str]:
        """Get preview of a specific chunk"""
        for chunk in self.last_retrieved_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.content
        return None
    
    def explain_retrieval(self) -> str:
        """Explain what was retrieved and why"""
        if not self.last_retrieved_chunks:
            return "No chunks were retrieved."
        
        explanation = f"Retrieved {len(self.last_retrieved_chunks)} relevant chunks:\n\n"
        
        for i, chunk in enumerate(self.last_retrieved_chunks[:3], 1):
            explanation += f"{i}. **{chunk.source}** (Similarity: {chunk.similarity_score:.2%})\n"
            preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
            explanation += f"   Preview: {preview}\n\n"
        
        return explanation