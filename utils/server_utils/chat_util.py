import os
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union
from utils.rag_utils import MODEL_COST_PER_1K_TOKENS
from utils.rag_utils.agent_util import GPTAgent, ClaudeAgent
from utils.rag_utils.rag_util import load_index_and_metadata


class ChatInput(BaseModel):
    query: str
    index_name: str
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[Union[int, None]] = 5
    max_lexical_retrieval_chunks: Optional[Union[int, None]] = 1
    max_tokens: Optional[int] = 512


class ChatOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class ChatUtil:

    def __init__(self, indices_dir: str = None, index_name: str = None, server_logger=None):
        """
        Initializes the ChatUtil object by loading the FAISS index and metadata from the specified directory.

        :param indices_dir: The directory where the indices are stored. Default is None.
        :param index_name: The name of the index to be loaded. Default is None.
        :param server_logger: Serve logger. Optional.
        """
        index_meta = load_index_and_metadata(index_directory=os.path.join(indices_dir, index_name))
        faiss_index = index_meta["faiss_index"]
        metadata_dict = index_meta["metadata_dict"]
        lexical_index = index_meta["lexical_index"]
        self.gpt_agent = GPTAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None,
            server_logger=server_logger
        )
        self.claude_agent = ClaudeAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None,
            server_logger=server_logger
        )

    def generate_text(
        self, query: str = None, temperature: float = None, embedding_model: str = None, llm_model: str = None,
        max_semantic_retrieval_chunks: Union[int, None] = None, max_lexical_retrieval_chunks: Union[int, None] = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Generates a prediction response based on the input query and the provided parameters.

        :param query: The input text query for which to generate a response. Default is None.
        :param temperature: Controls the randomness of the output (higher value means more random). Default is None.
        :param embedding_model: The name of the embedding model to use. Default is None.
        :param llm_model: The name of the language model to use. Default is None.
        :param max_semantic_retrieval_chunks: The maximum number of chunks for semantic retrieval. Default is None.
        :param max_lexical_retrieval_chunks: The maximum number of chunks for lexical retrieval. Default is None.
        :param max_tokens: The maximum numbers of tokens a model should generate. Default is None.
        :return: A dictionary containing the generated text and other relevant information.
        """
        if 'gpt' in llm_model:
            query_agent = self.gpt_agent
        elif 'claude' in llm_model:
            query_agent = self.claude_agent
        else:
            raise ValueError(
                f"Unknown model: {llm_model}. Please provide a valid LLM model name."
                "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
            )
        return query_agent(
            query=query,
            temperature=temperature,
            embedding_model_name=embedding_model,
            stream=False,
            llm_model=llm_model,
            num_chunks=max_semantic_retrieval_chunks,
            lexical_search_k=max_lexical_retrieval_chunks,
            max_tokens=max_tokens
        )
