import os
import re
import ray
import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi


@ray.remote
class FaissIndexActor:

    def __init__(self, embedding_dim):
        """
        Initialize FAISS IndexFlatIP and metadata dictionary.
        :param embedding_dim: Embedding model dimension
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata_dict = {}

    @staticmethod
    def normalize_embedding(embedding):
        """
        Normalize embedding to unit vector.
        :param embedding: Numpy array
        :return Normalized numpy array
        """
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def add_embeddings_and_metadata(self, embeddings, texts, sources, embedding_model):
        """
        Add embeddings to FAISS index and metadata to metadata dictionary.
        :param embeddings: List[float numbers]
        :param texts: List of strings
        :param sources: List of strings
        :param embedding_model: Name of the embedding model
        :return
        """
        current_index = self.index.ntotal
        for text, source, emb in zip(texts, sources, embeddings):
            normalized_emb = self.normalize_embedding(emb)
            self.index.add(np.array([normalized_emb]))
            self.metadata_dict[current_index] = {
                'text': text, 'source': source, 'embedding_model': embedding_model[0]
            }
            current_index += 1

    def save_lexical_index(self, lexical_index_path):
        """
        Prepare and Save lexical index.
        """
        # Prepare chunks
        chunks = [i['text'] for i in self.metadata_dict.values()]

        # BM25 index
        texts = [re.sub(r"[^a-zA-Z0-9]", " ", chunk).lower().split() for chunk in chunks]
        lexical_index = BM25Okapi(texts)

        # Save index as pickle file
        with open(lexical_index_path, 'wb') as f:
            pickle.dump(lexical_index, f)

    def save_index(self, index_directory):
        """
        Save faiss index and metadata to index_dir.
        :param index_directory: Save index and metadata dict to this location.
        """
        index_path = os.path.join(index_directory, 'faiss.index')
        lexical_index_path = os.path.join(index_directory, 'lexical.index')
        metadata_path = os.path.join(index_directory, 'index.metadata')
        faiss.write_index(self.index, index_path)
        self.save_lexical_index(lexical_index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_dict, f)


class StoreResults:
    def __init__(self, faiss_actor):
        self.faiss_actor = faiss_actor

    def __call__(self, batch):
        ray.get(
            self.faiss_actor.add_embeddings_and_metadata.remote(
                batch["embeddings"], batch["text"], batch["source"], batch["embedding_model"]
            )
        )
        return {}


