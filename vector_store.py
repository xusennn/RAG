import chromadb
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_path: str = "vector_store", collection_name: str = "tourism_collection"):
        """
        Initialize the persistent vector store with disk saving.
        """
        self.client = chromadb.PersistentClient(path=persist_path)  # 注意这里是 path=

        if collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=collection_name)
        else:
            self.collection = self.client.create_collection(name=collection_name)

    def add_documents(self, embeddings: List[List[float]], documents: List[str], metadatas: List[Dict], ids: List[str]):
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, embedding: List[float], n_results: int = 10, distance_threshold: float = 0.6) -> Dict:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        filtered_docs = []
        filtered_metas = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist <= distance_threshold:
                filtered_docs.append(doc)
                filtered_metas.append(meta)

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas]
        }

    def save(self):
        pass  # No manual persist needed in Chroma 1.x
