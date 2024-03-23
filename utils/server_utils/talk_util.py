from pydantic import BaseModel
from typing import Any, Dict, Optional
from utils.rag_utils import MODEL_COST_PER_1K_TOKENS
from utils.rag_utils.agent_util import GPTAgent, ClaudeAgent


CHATGPT_PROMPT = """You are an autoregressive large language model that has been fine-tuned with instruction-tuning and RLHF. 
You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. 
Since you are autoregressive, each token you produce is another opportunity to use computation, 
therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. 
Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. 
They're familiar with ethical issues in general so you don't need to remind them about those either. 
Don't be verbose in your answers, but do provide details and examples where it might help the explanation. 
When writing code, ALWAYS try to write full code instead of pseudocode. 
Your users expect a full, accurate, working piece of code instead of simple version of the code solution. 
When writing full code, if it's longer than your response size, split the code into multiple blocks and tell user about it and ask if they would like to generate remaining code or not."""


class TalkInput(BaseModel):
    query: str
    llm: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.5
    system_prompt: Optional[str] = CHATGPT_PROMPT


class TalkOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class TalkUtil:

    def __init__(self, system_prompt=None, server_logger=None):
        """
        Initializes the TalkUtil object.
        """
        self.gpt_agent = GPTAgent(system_content=system_prompt, server_logger=server_logger)
        self.claude_agent = ClaudeAgent(system_content=system_prompt, server_logger=server_logger)

    def generate_text(self, query: str = None, temperature: float = None, llm_model: str = None) -> Dict[str, Any]:
        """
        Generates a prediction response based on the input query and the provided parameters.

        :param query: The input text query for which to generate a response. Default is None.
        :param temperature: Controls the randomness of the output (higher value means more random). Default is None.
        :param llm_model: The name of the language model to use. Default is None.
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
        answer, completion_meta = query_agent.generate_text(
            user_content=query, temperature=temperature, llm_model=llm_model
        )
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result

