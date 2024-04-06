import os
import re
import faiss
import pickle
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi


class FaissIndex:

    def __init__(self, embedding_dim: int):
        """
        Initializes the FaissIndex object with a specific embedding dimension. It sets up an FAISS IndexFlatIP
        for efficient similarity search in high-dimensional spaces and initializes a metadata dictionary to store
        additional information about each indexed embedding.

        :param embedding_dim: Dimensionality of the embeddings that will be added to the index.
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata_dict = {}

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        Normalizes an embedding vector to have a unit norm. This normalization is often required before
        using the embedding vector for similarity search in FAISS to ensure consistent performance.

        :param embedding: A numpy array representing the embedding vector to be normalized.
        :return: A normalized numpy array of the embedding vector.
        """
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def add_embeddings_and_metadata(
            self, embeddings: List[np.ndarray], texts: List[str], sources: List[str], embedding_model: str
    ):
        """
        Adds embeddings to the FAISS index and corresponding metadata to the metadata dictionary. Each embedding is
        normalized before being added to the index. Metadata for each embedding, including the original text, source,
        and embedding model name, is stored in a dictionary.

        :param embeddings: A list of embedding vectors to be added to the index.
        :param texts: A list of original texts corresponding to each embedding.
        :param sources: A list of source identifiers corresponding to each embedding.
        :param embedding_model: The name of the embedding model used to generate the embeddings.
        """
        current_index = self.index.ntotal
        for text, source, emb in zip(texts, sources, embeddings):
            normalized_emb = self.normalize_embedding(emb)
            self.index.add(np.array([normalized_emb]))
            self.metadata_dict[current_index] = {
                'text': text, 'source': source, 'embedding_model': embedding_model[0]
            }
            current_index += 1

    def save_lexical_index(self, lexical_index_path: str):
        """
        Prepares a lexical index using the BM25 algorithm from the texts in the metadata dictionary, and saves
        this lexical index to disk as a pickle file. This allows for efficient keyword-based search alongside the
        similarity search provided by the FAISS index.

        :param lexical_index_path: The file path where the lexical index will be saved.
        """
        # Prepare chunks
        chunks = [i['text'] for i in self.metadata_dict.values()]

        # BM25 index
        texts = [re.sub(r"[^a-zA-Z0-9]", " ", chunk).lower().split() for chunk in chunks]
        lexical_index = BM25Okapi(texts)

        # Save index as pickle file
        with open(lexical_index_path, 'wb') as f:
            pickle.dump(lexical_index, f)

    def save_index(self, index_directory: str):
        """
        Saves the FAISS index, the lexical index, and the metadata dictionary to disk. The FAISS index is saved
        in a binary format specific to FAISS, while the lexical index and metadata dictionary are saved as pickle files.

        :param index_directory: The directory where the index files and metadata dictionary will be saved.
        """
        index_path = os.path.join(index_directory, 'faiss.index')
        lexical_index_path = os.path.join(index_directory, 'lexical.index')
        metadata_path = os.path.join(index_directory, 'index.metadata')
        faiss.write_index(self.index, index_path)
        self.save_lexical_index(lexical_index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_dict, f)

