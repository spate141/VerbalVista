from pydantic import BaseModel
from typing import Any, Dict, Optional
from utils.rag_utils.agent_util import QueryAgent


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


class TalkOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class TalkUtil:

    def __init__(self):
        """
        Initializes the TalkUtil object.
        """
        self.query_agent = QueryAgent(system_content=CHATGPT_PROMPT)

    def generate_text(self, query: str = None, temperature: float = None, llm_model: str = None) -> Dict[str, Any]:
        """
        Generates a prediction response based on the input query and the provided parameters.

        :param query: The input text query for which to generate a response. Default is None.
        :param temperature: Controls the randomness of the output (higher value means more random). Default is None.
        :param llm_model: The name of the language model to use. Default is None.
        :return: A dictionary containing the generated text and other relevant information.
        """
        answer, completion_meta = self.query_agent.generate_text(
            user_content=query, temperature=temperature, llm_model=llm_model
        )
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result

