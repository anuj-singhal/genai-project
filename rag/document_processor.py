"""
Document Processor for RAG functionality
Handles document loading, parsing, and chunking
"""

import os
import hashlib
import tempfile
import platform
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import streamlit as st

# Document processing imports
try:
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
        UnstructuredMarkdownLoader
    )
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter
    )
except ImportError:
    st.error("Please install langchain: pip install langchain")

@dataclass
class Document:
    """Document data class"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_id: str = None

@dataclass
class ProcessedDocument:
    """Processed document with chunks"""
    doc_id: str
    filename: str
    chunks: List[Document]
    total_chunks: int
    total_tokens: int
    file_type: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Process documents for RAG pipeline"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.docx': 'docx',
        '.doc': 'docx',
        '.md': 'markdown'
    }
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 chunking_strategy: str = "token"):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of chunks in tokens or characters
            chunk_overlap: Overlap between chunks
            chunking_strategy: "token" or "character" based chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self._setup_text_splitter()
    
    def _setup_text_splitter(self):
        """Setup the text splitter based on strategy"""
        if self.chunking_strategy == "token":
            # Token-based splitting for more accurate control
            try:
                from langchain.text_splitter import TokenTextSplitter
                self.text_splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            except:
                # Fallback to character splitter
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size * 4,  # Approximate tokens to chars
                    chunk_overlap=self.chunk_overlap * 4,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
    
    def generate_doc_id(self, content: str, filename: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{filename}_{len(content)}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def process_uploaded_file(self, uploaded_file) -> ProcessedDocument:
        """
        Process an uploaded file into chunks
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            ProcessedDocument object with chunks
        """
        try:
            # Get file extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            file_type = self.SUPPORTED_EXTENSIONS[file_extension]
            
            # Create temp directory if it doesn't exist
            import tempfile
            import platform
            
            # Use appropriate temp directory for the OS
            if platform.system() == 'Windows':
                temp_dir = tempfile.gettempdir()
            else:
                temp_dir = '/tmp'
            
            # Ensure temp directory exists
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded file temporarily with a unique name
            temp_filename = f"upload_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}_{uploaded_file.name}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load document based on type
            content = self._load_document(temp_path, file_type)
            
            # Generate document ID
            doc_id = self.generate_doc_id(content, uploaded_file.name)
            
            # Split into chunks
            chunks = self._create_chunks(content, doc_id, uploaded_file.name)
            
            # Calculate tokens (approximate)
            total_tokens = sum(len(chunk.content) // 4 for chunk in chunks)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass  # Don't fail if cleanup fails
            
            return ProcessedDocument(
                doc_id=doc_id,
                filename=uploaded_file.name,
                chunks=chunks,
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                file_type=file_type,
                metadata={
                    "file_size": uploaded_file.size,
                    "processing_strategy": self.chunking_strategy
                }
            )
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return None
    
    def _load_document(self, file_path: str, file_type: str) -> str:
        """Load document content based on file type"""
        try:
            if file_type == 'pdf':
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                return content
                
            elif file_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif file_type == 'docx':
                import docx2txt
                return docx2txt.process(file_path)
                
            elif file_type == 'markdown':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except ImportError as e:
            st.error(f"Missing dependency for {file_type}: {str(e)}")
            st.info("Install required packages: pip install PyPDF2 python-docx docx2txt")
            return ""
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return ""
    
    def _create_chunks(self, content: str, doc_id: str, filename: str) -> List[Document]:
        """Split content into chunks"""
        chunks = []
        
        # Split text
        text_chunks = self.text_splitter.split_text(content)
        
        # Create Document objects
        for i, chunk_text in enumerate(text_chunks):
            chunk = Document(
                content=chunk_text,
                metadata={
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "doc_id": doc_id
                },
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_multiple_files(self, uploaded_files) -> List[ProcessedDocument]:
        """Process multiple uploaded files"""
        processed_docs = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            processed_doc = self.process_uploaded_file(uploaded_file)
            
            if processed_doc:
                processed_docs.append(processed_doc)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        
        return processed_docs
    
    def get_document_stats(self, processed_docs: List[ProcessedDocument]) -> Dict:
        """Get statistics about processed documents"""
        total_chunks = sum(doc.total_chunks for doc in processed_docs)
        total_tokens = sum(doc.total_tokens for doc in processed_docs)
        
        return {
            "total_documents": len(processed_docs),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "average_chunks_per_doc": total_chunks / len(processed_docs) if processed_docs else 0,
            "document_types": list(set(doc.file_type for doc in processed_docs))
        }