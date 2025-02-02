from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import os
import time
from io import BytesIO

class PDFEmbeddingProcessor:
    def __init__(self, pdf_file: str, model_name: str, ttl: int = 300):
        self.pdf_file = pdf_file
        self.ttl = ttl
        self.embeddings = None
        print('model name:')
        print(model_name)
        self.model = OllamaEmbeddings(model=model_name)

    def _load_pdf(self):
        loader = PyPDFLoader(self.pdf_file)
        documents = loader.load()
        return documents
    
    def _chunk_documents(self, documents, chunk_size=1000):
        text_chunks = []
        for doc in documents:
            text = doc.page_content
            for i in range(0, len(text), chunk_size):
                text_chunks.append(text[i:i + chunk_size])
        return text_chunks

    def _create_embeddings(self):
        documents = self._load_pdf()
        # print('docs')
        # print(documents)
        self.embeddings = self.model.embed_documents(self._chunk_documents(documents))

 
    
    def _store_embeddings_temporarily(self):
        if self.embeddings is None:
            raise ValueError("Embeddings have not been created yet.")
        vector_store = FAISS()
        vector_store.add_documents(self.embeddings)
        self._set_ttl(vector_store)

    def _set_ttl(self, vector_store):
        def remove_store():
            time.sleep(self.ttl)
            vector_store.delete()
        import threading
        threading.Thread(target=remove_store).start()

    def process(self):
        self._create_embeddings()
        self._store_embeddings_temporarily()

# Example usage:
# with open("/path/to/pdf", "rb") as f:
#     processor = PDFEmbeddingProcessor(f, "model-name", 3600)
#     processor.process()