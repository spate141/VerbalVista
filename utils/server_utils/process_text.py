from pydantic import BaseModel
from typing import Any, Dict, Optional


class ProcessTextInput(BaseModel):
    text: str
    chunk_size: Optional[int] = 600
    chunk_overlap: Optional[int] = 30
    embedding_model: Optional[str] = "text-embedding-ada-002"
    save_to_one_file: Optional[bool] = False
    text_description: Optional[str] = ""


class ProcessTextOutput(BaseModel):
    index_name: str
    index_meta: Dict[str, Any]


class ProcessTextUtil:

    def __init__(self, indices_dir: str = None, document_dir: str = None):
        self.indices_dir = indices_dir
        self.document_dir = document_dir

    @staticmethod
    async def process_text(text_meta: Dict[str, Any]):
        """

        """
        return [{
            "file_name": text_meta['text'][:20].replace("/", "-").replace('.', '-').replace(' ', '_'),
            "extracted_text": text_meta['text'],
            "doc_description": text_meta['description']
        }]

