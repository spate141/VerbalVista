import os
from pydantic import BaseModel
from typing import Any, Dict, Optional
from utils.rag_utils.agent_util import QueryAgent
from utils.rag_utils.rag_util import load_index_and_metadata


class QuestionInput(BaseModel):
    query: str
    index_name: str
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[int] = 5
    max_lexical_retrieval_chunks: Optional[int] = 1


class QuestionOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class QueryUtil:

    def __init__(self, indices_dir: str = None, index_name: str = None):
        """
        Initializes the QueryUtil object by loading the FAISS index and metadata from the specified directory.

        :param indices_dir: The directory where the indices are stored. Default is None.
        :param index_name: The name of the index to be loaded. Default is None.
        """
        index_meta = load_index_and_metadata(index_directory=os.path.join(indices_dir, index_name))
        faiss_index = index_meta["faiss_index"]
        metadata_dict = index_meta["metadata_dict"]
        lexical_index = index_meta["lexical_index"]
        self.query_agent = QueryAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None
        )

    def generate_text(
            self, query: str = None, temperature: float = None, embedding_model: str = None, llm_model: str = None,
            max_semantic_retrieval_chunks: int = None, max_lexical_retrieval_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Generates a prediction response based on the input query and the provided parameters.

        :param query: The input text query for which to generate a response. Default is None.
        :param temperature: Controls the randomness of the output (higher value means more random). Default is None.
        :param embedding_model: The name of the embedding model to use. Default is None.
        :param llm_model: The name of the language model to use. Default is None.
        :param max_semantic_retrieval_chunks: The maximum number of chunks for semantic retrieval. Default is None.
        :param max_lexical_retrieval_chunks: The maximum number of chunks for lexical retrieval. Default is None.
        :return: A dictionary containing the generated text and other relevant information.
        """
        result = self.query_agent(
            query=query,
            temperature=temperature,
            embedding_model_name=embedding_model,
            stream=False,
            llm_model=llm_model,
            num_chunks=max_semantic_retrieval_chunks,
            lexical_search_k=max_lexical_retrieval_chunks
        )
        return result

