from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SummaryUtil:

    def __init__(self):
        pass

    @staticmethod
    def process_text(text, chunk_size: int = 4000, chunk_overlap: int = 50):
        """
        This will split and convert the original text into LangChain Document.
        """
        r_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\. )"], chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = r_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
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
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
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

    def summarize(self, chain=None, text=None, question=None, chat_history=None):
        """

        """
        docs = self.process_text(text)
        with get_openai_callback() as cb:
            summary = chain.run(docs)
        chat_history.append((question, summary))
        answer_meta = f"""Total tokens: {cb.total_tokens} (Prompt: {cb.prompt_tokens} + Completion: {cb.completion_tokens})
        Total requests: {cb.successful_requests}
        Total cost (USD): {round(cb.total_cost, 6)}"""
        return summary, answer_meta, chat_history

