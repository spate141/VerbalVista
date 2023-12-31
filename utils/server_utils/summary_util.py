import os
import re
from pydantic import BaseModel
from typing import Any, Dict, Optional
from utils.rag_utils.agent_util import QueryAgent
from utils.rag_utils.rag_util import load_index_and_metadata


class SummaryInput(BaseModel):
    index_name: str
    summary_sentences_per_topic: Optional[int] = 3
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[int] = 5
    max_lexical_retrieval_chunks: Optional[int] = 1


class SummaryOutput(BaseModel):
    summary: str
    completion_meta: Dict[str, Any]


class SummaryUtil:

    def __init__(self, indices_dir: str = None, index_name: str = None):
        """
        Get query agent.
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

    @staticmethod
    def get_topics(answer):
        return [re.sub(r'^\d+\.\s*', '', text) for text in answer.split('\n')]

    def summarize_text(
            self, summary_sentences_per_topic: int = None, temperature: float = None, embedding_model: str = None,
            llm_model: str = None, max_semantic_retrieval_chunks: int = None, max_lexical_retrieval_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Generate topical summary from text.
        """
        generate_content_topics_query = "Generate a list of very high level topics from this text."
        response = self.query_agent(
            query=generate_content_topics_query,
            temperature=temperature,
            embedding_model_name=embedding_model,
            stream=False,
            llm_model=llm_model,
            num_chunks=max_semantic_retrieval_chunks,
            lexical_search_k=max_lexical_retrieval_chunks
        )
        high_level_topics_tokens = response['completion_meta']['tokens']
        high_level_topics_cost = response['completion_meta']['cost']

        topical_result = []
        for topic in self.get_topics(response['answer']):
            topical_query = f'Generate a short summary from the text about "{topic}" in {summary_sentences_per_topic} sentences.'
            t_result = self.query_agent(
                query=topical_query, temperature=temperature, embedding_model_name=embedding_model, stream=False,
                llm_model=llm_model, num_chunks=max_semantic_retrieval_chunks,
                lexical_search_k=max_lexical_retrieval_chunks
            )
            topical_result.append({
                "topic": topic,
                "topical_summary": t_result['answer'],
                "topical_tokens": t_result['completion_meta']['tokens'],
                "topical_cost": t_result['completion_meta']['cost']
            })

        summary = '\n'.join([f"{i['topic']}: {i['topical_summary']}" for i in topical_result])
        completion_meta_tokens = [i['topical_tokens'] for i in topical_result] + [high_level_topics_tokens]
        summed_completion_meta_tokens = {
            key: sum(d[key] for d in completion_meta_tokens if key in d) for key in set().union(*completion_meta_tokens)
        }
        completion_meta_cost = [i['topical_cost'] for i in topical_result] + [high_level_topics_cost]
        summed_completion_meta_cost = {
            key: sum(d[key] for d in completion_meta_cost if key in d) for key in set().union(*completion_meta_cost)
        }
        result = {
            "summary": summary,
            "completion_meta": {
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "temperature": temperature,
                "tokens": {
                    "completion": summed_completion_meta_tokens["completion"],
                    "prompt": summed_completion_meta_tokens["prompt"],
                    "total": summed_completion_meta_tokens["total"],
                },
                "cost": {
                    "completion": summed_completion_meta_cost["completion"],
                    "prompt": summed_completion_meta_cost["prompt"],
                    "total": summed_completion_meta_cost["total"],
                }
            }
        }
        return result
