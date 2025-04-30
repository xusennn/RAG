from sentence_transformers import SentenceTransformer
from typing import List, Union

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            embedding = self.model.encode(texts)
            return embedding
        elif isinstance(texts, list):
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        else:
            raise ValueError("Input must be a string or a list of strings.")
