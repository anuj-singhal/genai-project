## Project Overview

On-Prem GenAI Workbench - A Streamlit-based self-service GenAI platform with direct LLM chat and RAG (Retrieval-Augmented Generation) capabilities using OpenAI models.

## Commands

```bash
# Run the application (opens at localhost:8501)
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (required)
$env:OPENAI_API_KEY="sk-..."    # PowerShell
set OPENAI_API_KEY=sk-...       # Windows CMD
export OPENAI_API_KEY="sk-..."  # macOS/Linux
```

## Architecture

**Three-tier structure:**
- **Presentation:** Streamlit UI with sidebar configuration
- **Application:** Unified LLM interface orchestrating AI interactions
- **Backend:** RAG pipeline with ChromaDB vector store

**Key Files:**
- `app.py` - Streamlit entry point
- `config.py` - Centralized configuration (models, RAG settings, token limits)
- `llm_interface.py` - Main UI orchestration, handles both direct LLM and RAG modes

**RAG Module (`rag/`):**
- `document_processor.py` - Document loading (PDF, TXT, DOCX, MD) and chunking
- `vector_store.py` - ChromaDB wrapper with OpenAI or Sentence Transformers embeddings
- `rag_agent.py` - RAG pipeline: query analysis → retrieval → ranking → context building → response

**Data:**
- `data/chroma_db/` - Persistent ChromaDB vector store
- `data/rag_documents/` - Sample documents for testing

## Key Patterns

**Session State:** Streamlit session state for conversation history and RAG components
```python
if 'key' not in st.session_state:
    st.session_state.key = default_value
```

**Streaming:** Generator-based response streaming from OpenAI API

**RAG Pipeline:** Query → Check if RAG needed → Retrieve chunks → Rank by relevance → Build context → Generate with sources

**Embedding Abstraction:** Pluggable backends (OpenAI embeddings or local Sentence Transformers)

## Extension Points

- **New personas:** Add to `LLMInterface._load_example_prompt()` in `llm_interface.py`
- **New document formats:** Add to `DocumentProcessor.SUPPORTED_EXTENSIONS` in `rag/document_processor.py`
- **New embedding models:** Add to `Config.EMBEDDING_MODELS` in `config.py`
- **RAG ranking logic:** Modify `RAGAgent.rank_chunks()` in `rag/rag_agent.py`
- **Configuration options:** Update `config.py` and `LLMInterface._render_sidebar()`

## Testing

Manual testing scenarios in `examples/`:
- `examples/direct_llm.md` - 8 personas with test scenarios
- `examples/rag_test/quick_test_scenario.md` - RAG test questions by complexity

No automated test framework currently configured.
