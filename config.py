"""
Configuration file for On-Prem GenAI Workbench
"""

import os
from typing import List, Dict

class Config:
    """Configuration class for application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Available OpenAI Models
    OPENAI_MODELS: List[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    # Default Model
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Chat Configuration
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    TOP_P = 1.0
    MAX_TEMPERATURE = 1.0  # Maximum temperature value
    
    # UI Configuration
    CHAT_INPUT_HEIGHT = 100
    SYSTEM_PROMPT_HEIGHT = 150
    
    # Default System Prompt
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, accurate, and concise responses. 
If you're unsure about something, acknowledge it. Be professional and friendly in your interactions."""
    
    # Default Initial Knowledge/Context
    DEFAULT_INITIAL_KNOWLEDGE = ""  # Empty by default, user can add context
    
    # Future configurations for other scenarios
    # ChromaDB Configuration (for Scenario 2)
    CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    CHUNKING_STRATEGY = "token"  # "token" or "character"
    
    # RAG Configuration
    DEFAULT_TOP_K = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.3
    MAX_CONTEXT_TOKENS = 2000
    
    # Embedding Models
    EMBEDDING_MODELS = {
        "openai": {
            "text-embedding-ada-002": "OpenAI Ada v2 (Paid, High Quality)",
            "text-embedding-3-small": "OpenAI v3 Small (Paid, Fast)",
            "text-embedding-3-large": "OpenAI v3 Large (Paid, Best Quality)"
        },
        "sentence-transformer": {
            "all-MiniLM-L6-v2": "MiniLM (Free, Fast, 384 dim)",
            "all-mpnet-base-v2": "MPNet (Free, High Quality, 768 dim)",
            "all-MiniLM-L12-v2": "MiniLM L12 (Free, Balanced, 384 dim)",
            "paraphrase-MiniLM-L6-v2": "Paraphrase MiniLM (Free, Fast, 384 dim)"
        }
    }
    
    DEFAULT_EMBEDDING_TYPE = "openai"  # Changed to OpenAI by default for easier setup
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's most cost-effective model
    
    # DuckDB Configuration (for Scenario 3)
    DUCKDB_PATH = "./data/workbench.db"
    
    # Embedding Model (for Scenario 2)
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate if OpenAI API key is set"""
        return bool(cls.OPENAI_API_KEY)
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get information about a specific model"""
        model_info = {
            "gpt-4o-mini": {"context_window": 128000, "description": "Most cost-effective for most tasks"},
            "gpt-4o": {"context_window": 128000, "description": "High performance multimodal model"},
            "gpt-4-turbo": {"context_window": 128000, "description": "Latest GPT-4 Turbo model"},
            "gpt-4": {"context_window": 8192, "description": "Original GPT-4 model"},
            "gpt-3.5-turbo": {"context_window": 16385, "description": "Fast and cost-effective"},
            "gpt-3.5-turbo-16k": {"context_window": 16385, "description": "Extended context window"}
        }
        return model_info.get(model_name, {"context_window": 4096, "description": "Model information not available"})