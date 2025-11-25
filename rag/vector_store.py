"""
Vector Store Manager for ChromaDB operations
Handles embeddings storage and retrieval with both OpenAI and Sentence Transformers
"""

import os
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
import hashlib
import time
chromadb.api.client.SharedSystemClient.clear_system_cache()

@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    chunk_id: str
    source: str

class VectorStore:
    """Manage vector store operations with ChromaDB"""
    
    def __init__(self, 
                 persist_directory: str = "./data/chroma_db",
                 collection_name: str = "rag_documents"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self._init_chromadb()
        
        # Embedding model (will be set based on user choice)
        self.embedding_model = None
        self.embedding_type = None
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            self.collection = None
    
    def set_embedding_model(self, embedding_type: str, model_name: str = None, api_key: str = None):
        """
        Set the embedding model
        
        Args:
            embedding_type: "openai" or "sentence-transformer"
            model_name: Model name for embeddings
            api_key: API key for OpenAI (if using OpenAI embeddings)
        """
        self.embedding_type = embedding_type
        
        if embedding_type == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
            self.embedding_model = model_name or "text-embedding-ada-002"
            
        elif embedding_type == "sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer
                model_name = model_name or "all-MiniLM-L6-v2"
                with st.spinner(f"Loading {model_name} model (this may take a moment on first use)..."):
                    self.embedding_model = SentenceTransformer(model_name)
                st.success(f"✅ Loaded {model_name} for embeddings")
            except ImportError:
                st.error("""
                ❌ Sentence Transformers not installed!
                
                To use free local embeddings, please install:
                ```
                pip install sentence-transformers
                ```
                
                Or use OpenAI embeddings instead (requires API credits).
                """)
                raise ValueError("sentence-transformers package not installed. Install it or use OpenAI embeddings.")
            except Exception as e:
                st.error(f"Error loading sentence transformer model: {str(e)}")
                st.info("Try using OpenAI embeddings or install sentence-transformers properly.")
                raise
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not set. Call set_embedding_model first.")
        
        embeddings = []
        
        if self.embedding_type == "openai":
            # OpenAI embeddings
            try:
                # Process in batches to avoid rate limits
                batch_size = 20
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    # Small delay to avoid rate limits
                    if i + batch_size < len(texts):
                        time.sleep(0.1)
                        
            except Exception as e:
                st.error(f"Error generating OpenAI embeddings: {str(e)}")
                raise
                
        elif self.embedding_type == "sentence-transformer":
            # Sentence Transformer embeddings
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 10
                ).tolist()
            except Exception as e:
                st.error(f"Error generating sentence transformer embeddings: {str(e)}")
                raise
        
        return embeddings
    
    def add_documents(self, documents: List[Any], processed_docs: List[Any]) -> bool:
        """
        Add documents to vector store
        
        Args:
            documents: List of Document objects with chunks
            processed_docs: List of ProcessedDocument objects
            
        Returns:
            Success status
        """
        if not self.collection:
            st.error("ChromaDB collection not initialized")
            return False
        
        try:
            all_chunks = []
            all_ids = []
            all_metadatas = []
            
            # Collect all chunks from processed documents
            for proc_doc in processed_docs:
                for chunk in proc_doc.chunks:
                    all_chunks.append(chunk.content)
                    all_ids.append(chunk.chunk_id)
                    all_metadatas.append({
                        **chunk.metadata,
                        "filename": proc_doc.filename,
                        "file_type": proc_doc.file_type
                    })
            
            if not all_chunks:
                st.warning("No chunks to add to vector store")
                return False
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings for {len(all_chunks)} chunks..."):
                embeddings = self.generate_embeddings(all_chunks)
            
            # Add to ChromaDB
            with st.spinner("Adding to vector store..."):
                self.collection.add(
                    embeddings=embeddings,
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
            
            st.success(f"✅ Added {len(all_chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 5,
              similarity_threshold: float = 0.0) -> List[RetrievedChunk]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved chunks
        """
        if not self.collection:
            st.error("ChromaDB collection not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_chunks = []
            
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - results['distances'][0][i]
                    
                    if similarity_score >= similarity_threshold:
                        chunk = RetrievedChunk(
                            content=results['documents'][0][i],
                            metadata=results['metadatas'][0][i],
                            similarity_score=similarity_score,
                            chunk_id=results['ids'][0][i] if 'ids' in results else f"chunk_{i}",
                            source=results['metadatas'][0][i].get('source', 'Unknown')
                        )
                        retrieved_chunks.append(chunk)
            
            return retrieved_chunks
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks from a specific document
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Success status
        """
        if not self.collection:
            return False
        
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                st.success(f"Deleted {len(results['ids'])} chunks from document")
                return True
            else:
                st.warning("No chunks found for document")
                return False
                
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all documents from the collection"""
        if not self.collection:
            return False
        
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            st.success("Cleared all documents from vector store")
            return True
            
        except Exception as e:
            st.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if not self.collection:
            return {"status": "not_initialized"}
        
        try:
            count = self.collection.count()
            
            return {
                "status": "active",
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_type": self.embedding_type,
                "embedding_model": str(self.embedding_model) if self.embedding_model else None
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents in the collection"""
        if not self.collection:
            return []
        
        try:
            # Get all documents
            results = self.collection.get(
                include=["metadatas"]
            )
            
            # Extract unique documents
            documents = {}
            for metadata in results['metadatas']:
                doc_id = metadata.get('doc_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        'doc_id': doc_id,
                        'filename': metadata.get('source', 'Unknown'),
                        'chunks': 0
                    }
                if doc_id:
                    documents[doc_id]['chunks'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")
            return []