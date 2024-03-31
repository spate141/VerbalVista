from pydantic import BaseModel
from typing import Any, Dict, Optional
from utils.rag_utils.rag_util import do_some_chat_completion
from utils.rag_utils import MODEL_COST_PER_1K_TOKENS, NORMAL_SYS_PROMPT, CHATGPT_PROMPT
from utils.rag_utils.agent_util import GPTAgent, ClaudeAgent


class TalkInput(BaseModel):
    query: str
    llm: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.5
    system_prompt: Optional[str] = NORMAL_SYS_PROMPT
    max_tokens: Optional[int] = 512


class TalkOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class TalkUtil:

    def __init__(self, server_logger=None):
        """
        Initializes the TalkUtil object.
        """
        self.server_logger = server_logger

    def talk_with_llm(
        self, query: str = None, temperature: float = None, llm_model: str = None, max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Generates a prediction response based on the input query and the provided parameters.
        :param query: The input text query for which to generate a response. Default is None.
        :param temperature: Controls the randomness of the output (higher value means more random). Default is None.
        :param llm_model: The name of the language model to use. Default is None.
        :param max_tokens: Maximum numbers of tokens to generate in LLM response. Default is None.
        :return: A dictionary containing the generated text and other relevant information.
        """
        result = do_some_chat_completion(
            query=query, embedding_model="text-embedding-3-small", llm_model=llm_model, max_tokens=max_tokens,
            temperature=temperature, faiss_index=None, lexical_index=None, metadata_dict=None, reranker=None,
            server_logger=self.server_logger
        )
        return result

