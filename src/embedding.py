from langchain.embeddings import EmbeddingModel
from langchain.document_loaders import PDFLoader
from langchain.vectorstores import TemporaryVectorStore
import os

class PDFEmbeddingProcessor:
    def __init__(self, pdf_path: str, model_name: str, ttl: int = 3600):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.ttl = ttl
        self.embeddings = None 

    def load_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"The file {self.pdf_path} does not exist.")
        loader = PDFLoader(self.pdf_path)
        documents = loader.load()
        return documents

    def create_embeddings(self):
        documents = self.load_pdf()
        model = EmbeddingModel(self.model_name)
        self.embeddings = model.embed_documents(documents)

    def store_embeddings_temporarily(self):
        if self.embeddings is None:
            raise ValueError("Embeddings have not been created yet.")
        vector_store = TemporaryVectorStore(ttl=self.ttl)
        vector_store.add_documents(self.embeddings)

    def process(self):
        self.create_embeddings()
        self.store_embeddings_temporarily()

# Example usage:
# processor = PDFEmbeddingProcessor("/path/to/pdf", "model-name", 3600)
# processor.process()