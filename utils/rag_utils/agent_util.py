import os
import time
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from utils import log_error, log_debug
from utils.rag_utils import LLM_MAX_CONTEXT_LENGTHS, SYS_PROMPT, get_llm_token_cost_for_model
from utils.rag_utils.retrieval_util import get_query_embedding, do_lexical_search, do_semantic_search


class Agent:
    """
    The `Agent` class serves as a base class for agents that can generate text responses using language models.
    It includes methods for trimming text to a maximum context length, getting the number of tokens in a text,
    and placeholders for generating and preparing responses.
    The class also initializes components for lexical search, re-ranking, and semantic search.
    """
    def __init__(self, system_content=None, faiss_index=None, metadata_dict=None, lexical_index=None, reranker=None):

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

    def trim(self, text, max_context_length):
        return self.encoder.decode(self.encoder.encode(text)[:max_context_length])

    def get_num_tokens(self, text):
        return len(self.encoder.encode(text))

    @staticmethod
    def response_stream(response):
        raise "Sub-class should implement this!"

    def prepare_response(self, response, stream):
        if stream:
            raise "Sub-class should implement this!"
        else:
            raise "Sub-class should implement this!"

    def generate_text(
        self, llm_model, temperature=0.5, seed=42, stream=False, system_content="", user_content="",
        max_retries=1, retry_interval=60, embedding_model_name="", sources=None, max_tokens=None
    ):
        """Generate response from an LLM."""
        raise "Sub-class should implement this!"

    def __call__(
        self, query, num_chunks=5, stream=False, lexical_search_k=1, temperature=0.5, seed=42,
        embedding_model_name="EMBEDDING_MODEL_NAME", llm_model="LLM_MODEL_NAME", max_tokens=None
    ):
        raise "Sub-class should implement this!"


class GPTAgent(Agent):
    """
    The `GPTAgent` class inherits from `Agent` and is specialized in handling interactions with OpenAI's LLMs.
    It initializes an OpenAI client and implements methods for streaming responses, preparing responses,
    and generating text with retry logic.
    The `__call__` method orchestrates the process of embedding queries, performing semantic and lexical searches,
    potentially re-ranking results, and generating a final response based on a provided query.
    """
    def __init__(self, system_content=None, faiss_index=None, metadata_dict=None, lexical_index=None, reranker=None):
        super().__init__(system_content, faiss_index, metadata_dict, lexical_index, reranker)
        # Define OpenAI LLM model client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def response_stream(response):
        for chunk in response:
            yield chunk.choices[0].delta.content

    def prepare_response(self, response, stream):
        if stream:
            return self.response_stream(response)
        else:
            return response.choices[-1].message.content

    def generate_text(
        self, llm_model, temperature=0.5, seed=42, stream=False, system_content="", user_content="",
        max_retries=1, retry_interval=60, embedding_model_name="", sources=None, max_tokens=None
    ):
        """Generate response from an LLM."""
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
                log_error(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return None, None

    def __call__(
        self, query, num_chunks=5, stream=False, lexical_search_k=1, temperature=0.5, seed=42,
        embedding_model_name="text-embedding-3-small", llm_model="gpt-3.5-turbo", max_tokens=None
    ):
        log_debug(f'EMBEDDING_MODEL: {embedding_model_name} | LLM_MODEL: {llm_model} | TEMP: {temperature}')
        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=embedding_model_name)

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
            embedding_model_name=embedding_model_name,
            sources=sources,
            max_tokens=max_tokens,
        )
        # Result
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result


class ClaudeAgent(Agent):

    def __init__(self, system_content=None, faiss_index=None, metadata_dict=None, lexical_index=None, reranker=None):
        super().__init__(system_content, faiss_index, metadata_dict, lexical_index, reranker)
        # Define Anthropic LLM model client
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def response_stream(response):
        for chunk in response:
            if chunk.type == 'content_block_delta':
                yield chunk.delta.text

    def prepare_response(self, response, stream):
        if stream:
            return self.response_stream(response)
        else:
            return response.content[-1].text

    def generate_text(
        self, llm_model, temperature=0.5, seed=42, stream=False, system_content="", user_content="",
        max_retries=1, retry_interval=60, embedding_model_name="", sources=None, max_tokens=None
    ):
        """Generate response from an LLM."""
        retry_count = 0
        while retry_count <= max_retries:
            try:
                completion = self.client.messages.create(
                    model=llm_model,
                    temperature=temperature,
                    stream=stream,
                    system=system_content,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_content
                                }
                            ]
                        },
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
                log_error(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return None, None

    def __call__(
        self, query, num_chunks=5, stream=False, lexical_search_k=1, temperature=0.5, seed=42,
        embedding_model_name="text-embedding-3-small", llm_model="claude-3-opus-20240229", max_tokens=None
    ):
        log_debug(f'EMBEDDING_MODEL: {embedding_model_name} | LLM_MODEL: {llm_model} | TEMP: {temperature}')
        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=embedding_model_name)

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
            embedding_model_name=embedding_model_name,
            sources=sources,
            max_tokens=max_tokens
        )
        # Result
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result

