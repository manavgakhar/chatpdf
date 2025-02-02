
A streamlit application for chatting with PDF documents using:
- Ollama for LLM (run `ollama pull tinyllama` before starting)
- LangChain for document processing
- FAISS for vector similarity search
- In-memory vector store with configurable TTL for embeddings cleanup
## Quick Start
# Create environment and install dependencies
```
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install -r requirements.txt
```

# To run
```
streamlit run app.py
```

