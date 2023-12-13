from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from utils.data_parsing_utils.url_parser import process_url


class ProcessUrlsInput(BaseModel):
    urls: List[str]
    chunk_size: Optional[int] = 600
    chunk_overlap: Optional[int] = 30
    embedding_model: Optional[str] = "text-embedding-ada-002"
    save_to_one_file: Optional[bool] = False
    url_description: Optional[str] = ""


class ProcessUrlsOutput(BaseModel):
    index_name: str
    index_meta: Dict[str, Any]


class ProcessURLsUtil:

    def __init__(self, indices_dir : str = None, document_dir : str = None, reddit_util=None):
        self.indices_dir = indices_dir
        self.document_dir = document_dir
        self.reddit_util = reddit_util

    @staticmethod
    async def extract_text(urls_meta: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract text from the previously read file object based on file type.
        :param: urls_meta: List[Dict[str, Any]]: URLs metadata
        :return: List[Dict[str, Any]]: List of extracted text object dictionary.
        """
        extracted_texts = []
        for url_obj in urls_meta:
            url = url_obj['url']
            url_desc = url_obj['description']
            # if "reddit.com" in url:
            #     extracted_text = self.reddit_util.fetch_comments_from_url(url)
            #     extracted_text = ' '.join(extracted_text)
            # else:
            extracted_text = await process_url(url)
            extracted_texts.append({
                "file_name": url[8:].replace("/", "-").replace('.', '-'),
                "extracted_text": extracted_text,
                "doc_description": url_desc
            })
        return extracted_texts
