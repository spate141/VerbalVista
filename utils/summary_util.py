from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .logging_module import log_debug


class SummaryUtil:

    def __init__(self):
        pass

    @staticmethod
    def process_text(text, chunk_size: int = 4000, chunk_overlap: int = 50):
        """
        This will split and convert the original text into LangChain Document.
        """
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True
        )
        texts = r_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        log_debug(f'Text size: {len(text)}')
        log_debug(f'Total summarization docs created: {len(docs)}')
        return docs

    @staticmethod
    def initialize_summarization_chain(
            model_name: str = "gpt-3.5-turbo", temperature: float = 0.5,
            max_tokens: int = 512, chain_type: str = "map_reduce", verbose: bool = False
    ):
        """

        """
        llm = ChatOpenAI(
            temperature=temperature, model_name=model_name, max_tokens=max_tokens
        )
        map_prompt = """
        You are a highly skilled AI trained in language comprehension and summarization. 
        I would like you to read the following text and summarize it into a concise abstract paragraph. 
        Aim to retain the most important points, providing a coherent and readable summary that could help a person 
        understand the main points of the discussion without needing to read the entire text. 
        Please avoid unnecessary details or tangential points.
        "{text}"
        CONCISE SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt = """
        You are a proficient AI with a specialty in distilling information into key points.
        Write a concise summary of the following text delimited by triple backquotes.
        Based on the following text, identify and list the main points that were discussed or brought up. 
        These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. 
        Your goal is to provide a list that someone could read to quickly understand what was talked about.
        ```{text}```
        BULLET POINT SUMMARY:
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        summarization_chain = load_summarize_chain(
            llm, chain_type=chain_type, verbose=verbose,
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template
        )
        return summarization_chain

    def summarize(self, chain=None, text=None, chunk_size=4000, question=None, chat_history=None):
        """

        """
        docs = self.process_text(text, chunk_size=chunk_size)
        with get_openai_callback() as cb:
            summary = chain.run(docs)
        chat_history.append((question, summary))
        answer_meta = f"""Total tokens: {cb.total_tokens} (Prompt: {cb.prompt_tokens} + Completion: {cb.completion_tokens})
        Total requests: {cb.successful_requests}
        Total cost (USD): {round(cb.total_cost, 6)}"""
        return summary, answer_meta, chat_history

