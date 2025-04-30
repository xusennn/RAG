# main.py

from embedder import Embedder
from vector_store import VectorStore
from faiss_vector_store import FaissVectorStore
from prompt_builder import build_prompt
from generator import LocalGenerator


def main():
    # --- SELECT BACKEND ---
    use_faiss = True  # Set to False to use ChromaDB

    # 1. Load vector store
    if use_faiss:
        vector_store = FaissVectorStore(persist_path="faiss_index")
    else:
        vector_store = VectorStore(persist_path="vector_store")

    # 2. Initialize embedder and generator
    embedder = Embedder()
    generator = LocalGenerator()

    if use_faiss:
        print("FAISS index loaded.")
    else:
        print("Chroma VectorStore document count:", vector_store.collection.count())

    print("\nSystem ready. You can now ask travel-related questions!")
    print("-" * 60)

    while True:
        user_question = input("Your Question (or type 'exit' to quit): ")

        if user_question.lower() == 'exit':
            print("Goodbye!")
            break

        # 3. Encode user question
        query_embedding = embedder.encode(user_question)

        # 4. Retrieve relevant documents
        retrieval_results = vector_store.query(query_embedding, n_results=50)
        contexts = retrieval_results['documents'][0]
        print(contexts)

        if not contexts:
            print("Sorry, no relevant information found.")
            continue

        # 5. Build prompt
        prompt = build_prompt(contexts, user_question)

        # 6. Generate answer
        answer = generator.generate(prompt)

        # 7. Output
        print("\n--- Answer ---")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
