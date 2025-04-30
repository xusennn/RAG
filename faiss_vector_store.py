import faiss
import os
import pickle
import numpy as np
from typing import List, Dict

class FaissVectorStore:
    def __init__(self, persist_path: str = "faiss_index", dimension: int = 768):
        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, "faiss.index")
        self.meta_file = os.path.join(persist_path, "metadata.pkl")
        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadatas = []

        # load if exists
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.documents, self.metadatas = pickle.load(f)

    def add_documents(self, embeddings: List[List[float]], documents: List[str], metadatas: List[Dict], ids: List[str]):
        assert len(embeddings) == len(documents) == len(metadatas)
        np_embeddings = np.array(embeddings).astype("float32")
        self.index.add(np_embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def query(self, embedding: List[float], n_results: int = 10) -> Dict:
        if self.index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]]}

        np_embedding = np.array([embedding]).astype("float32")
        D, I = self.index.search(np_embedding, n_results)

        results_docs = [self.documents[i] for i in I[0] if i < len(self.documents)]
        results_meta = [self.metadatas[i] for i in I[0] if i < len(self.metadatas)]

        return {
            "documents": [results_docs],
            "metadatas": [results_meta]
        }

    def _save(self):
        os.makedirs(self.persist_path, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump((self.documents, self.metadatas), f)

    def save(self):
        self._save()
