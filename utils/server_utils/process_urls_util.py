import asyncio
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from utils.data_parsing_utils.url_parser import process_url, url_to_filename


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

    def __init__(self, indices_dir: str = None, document_dir: str = None, reddit_util=None):
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
        # Create a list of coroutine objects for each URL
        coroutines = [process_url(url_obj['url']) for url_obj in urls_meta]

        # Run all coroutine objects concurrently and wait for their results
        results = await asyncio.gather(*coroutines)

        # Combine results with metadata
        extracted_texts = [
            {
                "file_name": url_to_filename(urls_meta[i]['url']),
                "extracted_text": result,
                "doc_description": urls_meta[i]['description']
            } for i, result in enumerate(results)
        ]
        return extracted_texts
