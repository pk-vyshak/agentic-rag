from loguru import logger as lg
from qdrant_client import QdrantClient
from qdrant_client.models import  VectorParams, Distance
from sentence_transformers import SentenceTransformer

class BaseVectorStore:
    def __init__(self, collection_name="pdf_paragraphs"):
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        lg.info(f"Initialized BaseVectorStore with collection '{self.collection_name}'.")