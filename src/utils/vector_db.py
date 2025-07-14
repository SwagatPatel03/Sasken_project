# src/utils/vector_db.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

# Import or define the Change class
# from path.to.module import Change
class Change:
    pass

class VectorDB:
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("3gpp_changes")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_changes(self, changes: List[Change]):
        # Implementation placeholder
        pass
    
    def query_changes(self, query: str, top_k: int = 5):
        # Implementation placeholder
        pass