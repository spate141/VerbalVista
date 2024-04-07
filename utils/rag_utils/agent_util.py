import os
import time
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from typing import Tuple, Generator, Optional, Any, Dict, List
from dotenv import load_dotenv; load_dotenv(".env")
from utils import logger
from utils.rag_utils import LLM_MAX_CONTEXT_LENGTHS, SYS_PROMPT, get_llm_token_cost_for_model
from utils.rag_utils.retrieval_util import get_query_embedding, do_lexical_search, do_semantic_search


class Agent:
    """
    Initializes components essential for generating text responses using language models. This includes
    managing lexical search, re-ranking, semantic search, and handling of system and user content.
    """

    def __init__(
        self, system_content: Optional[str] = None, faiss_index: Optional[Any] = None,
        metadata_dict: Optional[dict] = None, lexical_index: Optional[Any] = None,
        reranker: Optional[Any] = None, server_logger=None
    ) -> None:
        """
        Initializes an agent with necessary components and configurations for text generation and search.
        :param system_content: Initial content or prompts used by the system for generating responses.
        :param faiss_index: FAISS index for semantic search.
        :param metadata_dict: Metadata dictionary containing information relevant to the semantic search.
        :param lexical_index: Lexical index for text-based search and matching.
        :param reranker: Reranker component for adjusting the order of search results based on additional criteria.
        :param server_logger: Logger for capturing server-side events and errors.
        """

        # Lexical search
        self.lexical_index = lexical_index

        # Re-ranker
        self.reranker = reranker

        # LLM
        self.client = None
        if system_content:
            self.system_content = system_content
        else:
            self.system_content = SYS_PROMPT
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Vectorstore
        self.faiss_index = faiss_index
        self.metadata_dict = metadata_dict

        # Server logger
        if server_logger:
            self.logger = server_logger
        else:
            self.logger = logger

    def trim(self, text: str, max_context_length: int) -> str:
        """
        Trims the text to a specified maximum context length.
        :param text: Text to be trimmed.
        :param max_context_length: Maximum number of tokens allowed in the context.
        :return: Trimmed text.
        """
        return self.encoder.decode(self.encoder.encode(text)[:max_context_length])

    def get_num_tokens(self, text: str) -> int:
        """
        Computes the number of tokens in the given text.
        :param text: Text for which to calculate the token count.
        :return: Number of tokens in the text.
        """
        return len(self.encoder.encode(text))

    @staticmethod
    def response_stream(response: Any) -> Generator[str, None, None]:
        """
        Streams responses from a language model. To be implemented by subclasses.
        """
        raise "Sub-class should implement this!"

    def prepare_response(self, response: Any, stream: bool) -> str:
        """
        Prepares the response from the language model for delivery. To be implemented by subclasses.
        :param response: The raw response from the language model.
        :param stream: Boolean flag indicating whether the response should be streamed.
        :return: Prepared text response for delivery.
        """
        if stream:
            raise "Sub-class should implement this!"
        else:
            raise "Sub-class should implement this!"

    def generate_text(
        self, llm_model: str, temperature: float = 0.5, seed: int = 42, stream: bool = False,
        system_content: str = "", user_content: str = "", max_retries: int = 1,
        retry_interval: int = 60, embedding_model_name: str = "", sources: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates text based on model parameters. To be implemented by subclasses.
        :param llm_model: The specific language model to use for text generation.
        :param temperature: Controls the randomness in the output generation. Higher values result in more varied outputs.
        :param seed: Random seed for generating deterministic outputs. Useful for debugging or repeatable outputs.
        :param stream: If True, the method streams the response, yielding partial outputs as they are generated.
        :param system_content: System-level content or prompts that guide the generation.
        :param user_content: User-specific content or queries that the response should address.
        :param max_retries: Number of retries in case of failures or rate limiting.
        :param retry_interval: Time to wait between retries, in seconds.
        :param embedding_model_name: Name of the embedding model used for semantic search or context embedding.
        :param sources: Optional list of sources or additional context information for the generation.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :return: A tuple containing the generated text and metadata about the generation process.
        """
        raise "Sub-class should implement this!"

    def __call__(
        self, query: str, num_chunks: int = 5, stream: bool = False, lexical_search_k: int = 1,
        temperature: float = 0.5, seed: int = 42, embedding_model_name: str = "EMBEDDING_MODEL_NAME",
        llm_model: str = "LLM_MODEL_NAME", max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes a query to generate a response. To be implemented by subclasses.
        """
        raise "Sub-class should implement this!"


class GPTAgent(Agent):
    """
    Specializes `Agent` for handling interactions with OpenAI's language models. Implements methods for
    generating text, streaming responses, and preparing responses.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a GPTAgent with configurations for OpenAI's client.
        """
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def response_stream(response: Any) -> Generator[str, None, None]:
        """
        Implements streaming of responses from OpenAI's language models.
        """
        for chunk in response:
            yield chunk.choices[0].delta.content

    def prepare_response(self, response: Any, stream: bool) -> str:
        """
        Prepares a response from OpenAI's language model for delivery.
        """
        if stream:
            return self.response_stream(response)
        else:
            return response.choices[-1].message.content

    def generate_text(
        self, llm_model: str, temperature: float = 0.5, seed: int = 42, stream: bool = False,
        system_content: str = "", user_content: str = "", max_retries: int = 1,
        retry_interval: int = 60, embedding_model_name: str = "", sources: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates text responses using OpenAI's language models with retry logic.
        """
        retry_count = 0
        while retry_count <= max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=llm_model,
                    seed=seed,
                    temperature=temperature,
                    stream=stream,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens
                )
                _completion_cost = get_llm_token_cost_for_model(
                    llm_model, completion.usage.completion_tokens, is_completion=True
                )
                _prompt_cost = get_llm_token_cost_for_model(
                    llm_model, completion.usage.prompt_tokens, is_completion=False
                )
                completion_meta = {
                    "llm_model": llm_model,
                    "embedding_model": embedding_model_name,
                    "temperature": temperature,
                    "tokens": {
                        "completion": completion.usage.completion_tokens,
                        "prompt": completion.usage.prompt_tokens,
                        "total": completion.usage.total_tokens,
                    },
                    "cost": {
                        "completion": _completion_cost,
                        "prompt": _prompt_cost,
                        "total": _completion_cost + _prompt_cost
                    },
                    "sources": sources
                }
                return self.prepare_response(response=completion, stream=stream), completion_meta

            except Exception as e:
                self.logger.error(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return None, None

    def __call__(
        self, query: str, num_chunks: int = 5, stream: bool = False, lexical_search_k: int = 1,
        temperature: float = 0.5, seed: int = 42, embedding_model_name: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo", max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates the process of generating a response to a query using OpenAI's language models.
        """
        try:
            indexed_data_embedding_model = self.metadata_dict[0]['embedding_model']
        except Exception as e:
            self.logger.error(e)
            indexed_data_embedding_model = embedding_model_name

        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=indexed_data_embedding_model)
        self.logger.debug(f'EMBEDDING_MODEL: {indexed_data_embedding_model} | '
                          f'LLM_MODEL: {llm_model} | '
                          f'TEMP: {temperature} | '
                          f'NUM_CHUNKS: {num_chunks}')

        # {id, distance, text, source}
        context_results = do_semantic_search(
            query_embedding, self.faiss_index, self.metadata_dict, k=num_chunks
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = do_lexical_search(self.lexical_index, query, self.metadata_dict, lexical_search_k)
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank
        if self.reranker:
            pass

        # Generate response
        context = '\n\n'.join([item["text"] for item in context_results])
        sources = [item["source"] for item in context_results]
        user_content = f"Query:\n```{query}```\n\nContext:\n```\n{context}\n\n```"
        max_context_length = LLM_MAX_CONTEXT_LENGTHS.get(llm_model, 4096)
        context_length = max_context_length - self.get_num_tokens(self.system_content)
        answer, completion_meta = self.generate_text(
            llm_model=llm_model,
            temperature=temperature,
            seed=seed,
            stream=stream,
            system_content=self.system_content,
            user_content=self.trim(user_content, context_length),
            embedding_model_name=indexed_data_embedding_model,
            sources=sources,
            max_tokens=max_tokens,
        )
        # Result
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result


class ClaudeAgent(Agent):
    """
    Inherits from `Agent` and specializes in handling interactions with Anthropic's Claude language models.
    Implements methods for streaming responses, preparing responses, and generating text with retry logic.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a ClaudeAgent with configurations for Anthropic's client.
        """
        super().__init__(**kwargs)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def response_stream(response: Any) -> Generator[str, None, None]:
        """
        Implements streaming of responses from Anthropic's Claude models.
        """
        for chunk in response:
            if chunk.type == 'content_block_delta':
                yield chunk.delta.text

    def prepare_response(self, response: Any, stream: bool) -> str:
        """
        Prepares a response from Anthropic's Claude model for delivery.
        """
        if stream:
            return self.response_stream(response)
        else:
            return response.content[-1].text

    def generate_text(
        self, llm_model: str, temperature: float = 0.5, seed: int = 42, stream: bool = False,
        system_content: str = "", user_content: str = "", max_retries: int = 1,
        retry_interval: int = 60, embedding_model_name: str = "", sources: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates text responses using Anthropic's Claude models with retry logic.
        """
        retry_count = 0
        while retry_count <= max_retries:
            try:
                completion = self.client.messages.create(
                    model=llm_model, temperature=temperature, stream=stream, system=system_content,
                    max_tokens=max_tokens, messages=[
                        {"role": "user", "content": [{"type": "text", "text": user_content}]},
                    ],
                )
                _completion_cost = get_llm_token_cost_for_model(
                    llm_model, completion.usage.output_tokens, is_completion=True
                )
                _prompt_cost = get_llm_token_cost_for_model(
                    llm_model, completion.usage.input_tokens, is_completion=False
                )
                completion_meta = {
                    "llm_model": llm_model,
                    "embedding_model": embedding_model_name,
                    "temperature": temperature,
                    "tokens": {
                        "completion": completion.usage.output_tokens,
                        "prompt": completion.usage.input_tokens,
                        "total": completion.usage.output_tokens + completion.usage.input_tokens,
                    },
                    "cost": {
                        "completion": _completion_cost,
                        "prompt": _prompt_cost,
                        "total": _completion_cost + _prompt_cost
                    },
                    "sources": sources
                }
                return self.prepare_response(response=completion, stream=stream), completion_meta

            except Exception as e:
                self.logger.error(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return None, None

    def __call__(
        self, query: str, num_chunks: int = 5, stream: bool = False, lexical_search_k: int = 1,
        temperature: float = 0.5, seed: int = 42, embedding_model_name: str = "text-embedding-3-small",
        llm_model: str = "claude-3-opus-20240229", max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates the process of generating a response to a query using Anthropic's Claude models.
        """
        try:
            indexed_data_embedding_model = self.metadata_dict[0]['embedding_model']
        except Exception as e:
            self.logger.error(e)
            indexed_data_embedding_model = embedding_model_name

        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=indexed_data_embedding_model)
        self.logger.debug(f'EMBEDDING_MODEL: {indexed_data_embedding_model} | '
                          f'LLM_MODEL: {llm_model} | '
                          f'TEMP: {temperature} | '
                          f'NUM_CHUNKS: {num_chunks}')

        # {id, distance, text, source}
        context_results = do_semantic_search(
            query_embedding, self.faiss_index, self.metadata_dict, k=num_chunks
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = do_lexical_search(self.lexical_index, query, self.metadata_dict, lexical_search_k)
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank
        if self.reranker:
            pass

        # Generate response
        context = '\n\n'.join([item["text"] for item in context_results])
        sources = [item["source"] for item in context_results]
        user_content = f"Query:\n```{query}```\n\nContext:\n```\n{context}\n\n```"
        max_context_length = LLM_MAX_CONTEXT_LENGTHS.get(llm_model, 4096)
        context_length = max_context_length - self.get_num_tokens(self.system_content)
        answer, completion_meta = self.generate_text(
            llm_model=llm_model,
            temperature=temperature,
            seed=seed,
            stream=stream,
            system_content=self.system_content,
            user_content=self.trim(user_content, context_length),
            embedding_model_name=indexed_data_embedding_model,
            sources=sources,
            max_tokens=max_tokens
        )
        # Result
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result
