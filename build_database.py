from data_loader import DataLoader
from text_processor import build_embedding_text
from embedder import Embedder
from vector_store import VectorStore
import os

def build_vector_store(data_path: str = "data.json", persist_path: str = "vector_store"):
    """
    Build the vector database and persist to disk.
    """
    # 1. Load raw data
    loader = DataLoader(data_path)
    data = loader.load_json()

    # 2. Build embedding texts
    texts = build_embedding_text(data)

    # 3. Generate embeddings
    embedder = Embedder()
    embeddings = embedder.encode(texts)

    # 4. Prepare ids and metadatas
    ids = [str(i) for i in range(len(texts))]
    metadatas = [{"country": item.get("country", ""), "attraction": item.get("attraction", "")} for item in data]

    # 5. Store into VectorStore with persistence
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    # 只初始化一次！
    store = VectorStore(persist_path=persist_path)

    # 加数据
    assert len(embeddings) == len(texts) == len(metadatas) == len(ids), \
        f"Length mismatch: embeddings={len(embeddings)}, texts={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)}"

    store.add_documents(embeddings, texts, metadatas, ids)

    # 添加后再检查一次collection数量
    print("After add, VectorStore document count:", store.collection.count())

    store.save()  # save可以是空的pass（1.x自动持久化）

    print(f"Vector store successfully built and saved at '{persist_path}'.")

if __name__ == "__main__":
    build_vector_store()
