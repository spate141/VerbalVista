import os
import re
from pydantic import BaseModel
from num2words import num2words
from typing import Any, Dict, Optional
from utils.rag_utils import MODEL_COST_PER_1K_TOKENS
from utils.rag_utils.agent_util import GPTAgent, ClaudeAgent
from utils.rag_utils.rag_util import load_index_and_metadata


class SummaryInput(BaseModel):
    index_name: str
    summary_sentences_per_topic: Optional[int] = 3
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[int] = 5
    max_lexical_retrieval_chunks: Optional[int] = 1
    max_tokens: Optional[int] = 512


class SummaryOutput(BaseModel):
    summary: str
    completion_meta: Dict[str, Any]


class SummaryUtil:

    def __init__(self, indices_dir: str = None, index_name: str = None):
        """
        Initializes a SummaryUtil instance by loading necessary indices and metadata,
        and setting up a GPTAgent for generating summary content.

        :param indices_dir: Directory where indices are stored.
        :param index_name: Name of the specific index to load.
        """
        index_meta = load_index_and_metadata(index_directory=os.path.join(indices_dir, index_name))
        faiss_index = index_meta["faiss_index"]
        metadata_dict = index_meta["metadata_dict"]
        lexical_index = index_meta["lexical_index"]
        self.gpt_agent = GPTAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None
        )
        self.claude_agent = ClaudeAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None
        )

    @staticmethod
    def get_topics(answer):
        """
        Extracts topics from an answer string, removing any leading numbering.

        :param answer: String containing newline-separated topics with optional numbering.
        :return: List of topic strings without numbering.
        """
        return [re.sub(r'^\d+\.\s*', '', text) for text in answer.split('\n')]

    def summarize_text(
        self, summary_sentences_per_topic: int = None, temperature: float = None, embedding_model: str = None,
        llm_model: str = None, max_semantic_retrieval_chunks: int = None, max_lexical_retrieval_chunks: int = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Generates a summary of text based on high-level topics extracted from the content.
        Each topic will have a summary of a specified number of sentences.

        :param summary_sentences_per_topic: Number of sentences per topic summary.
        :param temperature: Controls randomness in the generation process.
        :param embedding_model: Model used for semantic embeddings.
        :param llm_model: Large language model used for generating content.
        :param max_semantic_retrieval_chunks: Maximum chunks for semantic retrieval.
        :param max_lexical_retrieval_chunks: Maximum chunks for lexical retrieval.
        :param max_tokens: Maximum numbers of tokens to generate.
        :return: Dictionary containing the summary, topics, and metadata about the generation process.
        """
        common_params = {
            "temperature": temperature,
            "embedding_model_name": embedding_model,
            "stream": False,
            "llm_model": llm_model,
            "max_tokens": max_tokens
        }

        # Generate a list of topics from the given text
        if 'gpt' in llm_model:
            query_agent = self.gpt_agent
        elif 'claude' in llm_model:
            query_agent = self.claude_agent
        else:
            raise ValueError(
                f"Unknown model: {llm_model}. Please provide a valid LLM model name."
                "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
            )
        response = query_agent(
            query="Generate a list of high level topics discussed in this text. "
                  "Make sure the generated topics represent entirety of the text and are unique. "
                  "List the topics in chronological order of text.",
            num_chunks=15,
            lexical_search_k=0,
            **common_params
        )
        topics = self.get_topics(response['answer'])

        # For each topic generated above, generate `n` sentences summary
        topical_result = []
        tokens, costs = {"completion": 0, "prompt": 0, "total": 0}, {"completion": 0, "prompt": 0, "total": 0}
        for topic in topics:
            t_result = query_agent(
                query=f'Generate a very short summary from the text about "{topic}" in {num2words(summary_sentences_per_topic)} sentences.',
                num_chunks=max_semantic_retrieval_chunks,
                lexical_search_k=max_lexical_retrieval_chunks,
                **common_params
            )
            topical_result.append((topic, t_result['answer']))
            tokens = {key: tokens[key] + t_result['completion_meta']['tokens'][key] for key in tokens}
            costs = {key: costs[key] + t_result['completion_meta']['cost'][key] for key in costs}

        # Generate a final summary text and add tokens and cost meta
        summary = '\n\n'.join([
            f"{index}. {topic}: {summary}" for index, (topic, summary) in enumerate(topical_result, 1)
        ])
        tokens["completion"] += response['completion_meta']['tokens']["completion"]
        costs["completion"] += response['completion_meta']['cost']["completion"]

        return {
            "summary": summary,
            "completion_meta": {
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "temperature": temperature,
                "tokens": tokens,
                "cost": costs
            }
        }

