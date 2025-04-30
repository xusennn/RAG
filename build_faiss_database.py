from data_loader import DataLoader
from text_processor import build_embedding_text
from embedder import Embedder
from faiss_vector_store import FaissVectorStore
import os


def build_faiss_vector_store(data_path: str = "data.json", persist_path: str = "faiss_index"):
    loader = DataLoader(data_path)
    data = loader.load_json()

    texts = build_embedding_text(data)
    embedder = Embedder()
    embeddings = embedder.encode(texts)

    ids = [str(i) for i in range(len(texts))]
    metadatas = [{"country": item.get("country", ""), "attraction": item.get("attraction", "")} for item in data]

    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    store = FaissVectorStore(persist_path=persist_path)
    store.add_documents(embeddings, texts, metadatas, ids)

    print("FAISS vector store built and saved at:", persist_path)


if __name__ == "__main__":
    build_faiss_vector_store()