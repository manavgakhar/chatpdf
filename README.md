
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

# Demo
## Uploaded pdf of airline check-in pdf locally:
<img width="845" alt="Screenshot 2025-02-01 at 8 55 44 PM" src="https://github.com/user-attachments/assets/d52a0623-436c-4f4d-a049-ae3e13a68bb7" />
