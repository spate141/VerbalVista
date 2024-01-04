import os
import time
import logging
import argparse
from ray import serve
import multiprocessing
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, status, Path
from dotenv import load_dotenv; load_dotenv(".env")
from fastapi import UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

from utils.openai_utils import OpenAIWisperUtil
from utils.rag_utils.rag_util import index_data
from utils.data_parsing_utils import write_data_to_file
from utils.server_utils import (
    QueryUtil, ProcessTextUtil, ProcessURLsUtil, ProcessMultimediaUtil, ListIndicesUtil, DeleteIndexUtil,
    AuthUtil, SummaryUtil, ChatHistoryUtil
)
from utils.server_utils import (
    QuestionInput, ProcessTextInput, ProcessUrlsInput, ProcessMultimediaInput, SummaryInput
)
from utils.server_utils import (
    QuestionOutput, ProcessTextOutput, ProcessUrlsOutput, ProcessMultimediaOutput, ListIndicesOutput,
    DeleteIndexOutput, SummaryOutput, ChatHistoryOutput
)
from utils.data_parsing_utils.reddit_comment_parser import RedditSubmissionCommentsFetcher


app = FastAPI(
    title="Inference API for VerbalVista",
    description="🄻🄻🄼 + Your Data = ♥️",
    version="1.8",
)
auth_util = AuthUtil(
    valid_api_keys=os.getenv("VALID_API_KEYS", "").split(",")
)
origins = [
    "http://127.0.0.1:8000",  # Local
    # "https://myapp.example.com",  # Production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


class HealthCheck(BaseModel):
    status: str = "OK"


@serve.deployment()
@serve.ingress(app)
class VerbalVistaAssistantDeployment:
    """
    Deployment class for the VerbalVista Ray Assistant. This class initializes
    the necessary utilities and directories needed for processing queries and documents.
    """
    def __init__(self, logging_level=None):
        """
        Initializes the VerbalVista Assistant with necessary configurations.

        Args:

            logging_level (Optional[int]): Specifies the logging level for the application.
        """
        self.logger = logging.getLogger("ray.serve")
        self.logger.setLevel(logging_level)
        self.openai_wisper_util = OpenAIWisperUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.reddit_util = RedditSubmissionCommentsFetcher(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.tmp_audio_dir = 'data/tmp_audio_dir/'
        self.document_dir = 'data/documents/'
        self.indices_dir = 'data/indices/'
        self.chat_history_dir = 'data/chat_history/'

    @app.get(
        "/list/indices",
        tags=["list"],
        summary="Returns list of available indices.",
        response_model=ListIndicesOutput,
    )
    def get_indices(self, api_key: str = Depends(auth_util.get_api_key)) -> ListIndicesOutput:
        """
        Retrieves a list of all indices currently available in the system.

        This endpoint does not require any parameters and will return a `ListIndicesOutput` object containing an array of index names.

        Args:
            api_key (str): APIKeyHeader

        Returns:
            ListIndicesOutput: An object that includes an array of the names of all available indices.
        """
        start = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /get_indices")
        list_indices_util = ListIndicesUtil()
        result = list_indices_util.list_indices(indices_dir=self.indices_dir)
        end = time.time()
        self.logger.info(f"Finished /list/indices in {round((end - start) * 1000, 2)} ms")
        return ListIndicesOutput.parse_list(result)

    @app.delete(
        "/delete/{index_name}",
        tags=["delete"],
        summary="Delete given index.",
        response_model=DeleteIndexOutput,
    )
    def delete_index(
            self, index_name: str = Path(..., description="The ID of the index to be deleted"),
            api_key: str = Depends(auth_util.get_api_key)
    ) -> DeleteIndexOutput:
        """
        Deletes the specified index from the system.

        This endpoint will attempt to delete an index identified by the `index_name` parameter.
        If successful, it returns a `DeleteIndexOutput` object containing the result of the deletion.

        Args:
            index_name (str): The unique identifier of the index to be deleted.
            api_key (str): APIKeyHeader

        Returns:
            DeleteIndexOutput: An object that includes the status of the deletion operation and any additional information.
        """
        start = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /delete_index")
        delete_index_util = DeleteIndexUtil()
        result = delete_index_util.delete_index(index_dir=self.indices_dir, index_name=index_name)
        end = time.time()
        self.logger.info(f"Finished /delete/{index_name} in {round((end - start) * 1000, 2)} ms")
        return DeleteIndexOutput.model_validate(result)

    @app.get(
        "/chat/{index_name}",
        tags=["chat"],
        summary="Get chat history for given index.",
        response_model=ChatHistoryOutput,
    )
    def chat_history(
            self, index_name: str = Path(..., description="The ID of the index to get chat history for."),
            api_key: str = Depends(auth_util.get_api_key)
    ) -> ChatHistoryOutput:
        """
        Get the Q&A chat history for the specified index.

        This endpoint will attempt to get the chat history for an index identified by the `index_name` parameter.
        If successful, it returns a `ChatHistoryOutput` object containing the result of the chat.

        Args:
            index_name (str): The unique identifier of the index to be deleted.
            api_key (str): APIKeyHeader

        Returns:
            ChatHistoryOutput: An object that includes the chat history of the input index.
        """
        start = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /chat/{index_name}")
        chat_history_util = ChatHistoryUtil(chat_history_dir=self.chat_history_dir, index_name=index_name)
        result = chat_history_util.load_chat_history()
        end = time.time()
        self.logger.info(f"Finished /chat/{index_name} in {round((end - start) * 1000, 2)} ms")
        return ChatHistoryOutput.parse_list(result)

    @app.post(
        "/query",
        tags=["query"],
        summary="Use index and LLMs to generate response to user's question.",
        response_model=QuestionOutput,
    )
    def query(self, query: QuestionInput, api_key: str = Depends(auth_util.get_api_key)) -> QuestionOutput:
        """
        Processes a query and returns the predicted answer.

        Args:
            query (QuestionInput): The query input containing parameters for processing.
            api_key (str): APIKeyHeader

        Returns:
            QuestionOutput: The output containing the predicted answer.
        """
        start = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /query")
        query_util = QueryUtil(indices_dir=self.indices_dir, index_name=query.index_name)
        chat_history_util = ChatHistoryUtil(chat_history_dir=self.chat_history_dir, index_name=query.index_name)
        end = time.time()
        self.logger.info(f"Query agent initiated in {round((end - start) * 1000, 2)} ms")
        chat_history_util.save_chat(role="user", content=query.query, meta=None)
        result = query_util.generate_text(
            query=query.query, temperature=query.temperature, embedding_model=query.embedding_model,
            llm_model=query.llm, max_semantic_retrieval_chunks=query.max_semantic_retrieval_chunks,
            max_lexical_retrieval_chunks=query.max_lexical_retrieval_chunks
        )
        chat_history_util.save_chat(role="assistant", content=result['answer'], meta=result['completion_meta'])
        end = time.time()
        self.logger.info(f"Finished /query in {round((end - start) * 1000, 2)} ms")
        return QuestionOutput.model_validate(result)

    @app.post(
        "/summarize",
        tags=["query"],
        summary="Use LLMs to generate a summary from given index.",
        response_model=SummaryOutput,
    )
    def summarize(self, query: SummaryInput, api_key: str = Depends(auth_util.get_api_key)) -> SummaryOutput:
        """
        Summarize index.

        Args:
            query (SummaryInput): The query input containing parameters for processing.
            api_key (str): APIKeyHeader

        Returns:
            SummaryOutput: The output containing the summary.
        """
        start = time.time()
        self.logger.info(f"Request received for /summarize endpoint; API Key: {api_key}")
        summary_util = SummaryUtil(indices_dir=self.indices_dir, index_name=query.index_name)
        chat_history_util = ChatHistoryUtil(chat_history_dir=self.chat_history_dir, index_name=query.index_name)
        end = time.time()
        self.logger.info(f"Query agent initiated in {round((end - start) * 1000, 2)} ms")
        chat_history_util.save_chat(
            role="user",
            content=f"Summary for index: {query.index_name} with {query.summary_sentences_per_topic} sentences per topic.",
            meta=None
        )
        result = summary_util.summarize_text(
            summary_sentences_per_topic=query.summary_sentences_per_topic, temperature=query.temperature,
            embedding_model=query.embedding_model, llm_model=query.llm,
            max_semantic_retrieval_chunks=query.max_semantic_retrieval_chunks,
            max_lexical_retrieval_chunks=query.max_lexical_retrieval_chunks
        )
        chat_history_util.save_chat(role="assistant", content=result['summary'], meta=result['completion_meta'])
        end = time.time()
        self.logger.info(f"Finished /summarize in {round((end - start) * 1000, 2)} ms")
        return SummaryOutput.model_validate(result)

    @app.post(
        "/process/multimedia",
        tags=["process"],
        summary="Process documents and generate document index.",
        response_model=ProcessMultimediaOutput,
    )
    async def process_multimedia(
            self, files: List[UploadFile] = File(...),
            data: ProcessMultimediaInput = Depends(ProcessMultimediaInput.as_form),
            api_key: str = Depends(auth_util.get_api_key)
    ) -> ProcessMultimediaOutput:
        """
        Processes multimedia documents by indexing them and logs the operation.

        Supported files:

        ## Audio Files
        The endpoint supports the following audio file formats:
        - `.m4a` - MPEG-4 Audio, used mainly for music and other audio recordings.
        - `.mp3` - A popular audio format for music and audio streaming, known for its compression capabilities.
        - `.wav` - Waveform Audio File Format, used for uncompressed audio data.
        - `.webm` - A modern media format for web videos, can contain both audio and video, but also supports audio-only files.
        - `.mpga` - Often associated with MPEG-1 or MPEG-2 audio layer 3, another extension for MP3 files.

        ## Video Files
        Supported video file formats include:
        - `.mp4` - A digital multimedia container format commonly used to store video and audio.
        - `.mpeg` - A standard format for lossy compression of video and audio.

        ## Document Files
        The endpoint can process the following document file formats:
        - `.pdf` - Portable Document Format, used for presenting documents with text formatting and images.
        - `.docx` - Microsoft Word Open XML Format Document, the default file format for Word documents since Office 2007.
        - `.txt` - Standard text file format, containing unformatted text.
        - `.eml` - Email message saved to a file in the Internet Message Format protocol.

        Args:
            files (List[UploadFile]): The list of files to be processed.
            data (ProcessMultimediaInput): Additional data for processing the document.
            api_key (str): APIKeyHeader

        Returns:
            ProcessMultimediaOutput: The result of the document processing.
        """

        # Initialize Process Documents Util Class
        process_multimedia_util = ProcessMultimediaUtil(
            indices_dir=self.indices_dir, document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir,
            openai_wisper_util=self.openai_wisper_util
        )

        # (1) Read the file content
        start1 = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /process/multimedia")
        files_meta = await process_multimedia_util.read_files(files, file_description=data.file_description)
        end = time.time()
        self.logger.info(f"Finished reading `{len(files_meta)}` files in {round((end - start1) * 1000, 2)} ms")

        # (2) Process files content and extract text
        start = time.time()
        extracted_texts = process_multimedia_util.extract_text(files_meta=files_meta)
        end = time.time()
        self.logger.info(f"Text Extracted from `{len(files_meta)}` files in {round((end - start) * 1000, 2)} ms")

        # (3) Write extracted texts to tmp file
        start = time.time()
        tmp_document_save_path = write_data_to_file(
            document_dir=self.document_dir,
            full_documents=extracted_texts,
            single_file_flag=data.save_to_one_file,
        )
        end = time.time()
        self.logger.info(f"Extracted text saved to `{tmp_document_save_path}` in {round((end - start) * 1000, 2)} ms")

        # (4) Generate FAISS index
        start = time.time()
        document_directory = os.path.join(self.document_dir, tmp_document_save_path)
        index_directory = os.path.join(self.indices_dir, tmp_document_save_path)
        index_data(
            document_directory=document_directory,
            index_directory=index_directory,
            chunk_size=data.chunk_size,
            chunk_overlap=data.chunk_overlap,
            embedding_model=data.embedding_model
        )
        result = {"index_name": os.path.basename(index_directory)}
        end = time.time()
        self.logger.info(f"FAISS index saved to `{index_directory}` in {round((end - start) * 1000, 2)} ms")

        # (5) Construct a metadata dictionary from the processing data.
        _ = [file_meta.pop("file", None) for file_meta in files_meta]

        result["index_meta"] = {
            "files_meta": files_meta,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }
        end = time.time()
        self.logger.info(f"Finished /process/documents in {round((end - start1) * 1000, 2)} ms")
        return ProcessMultimediaOutput.model_validate(result)

    @app.post(
        "/process/urls",
        tags=["process"],
        summary="Process URLs and generate URLs index.",
        response_model=ProcessUrlsOutput,
    )
    async def process_urls(
            self, data: ProcessUrlsInput, api_key: str = Depends(auth_util.get_api_key)
    ) -> ProcessUrlsOutput:
        """
        Extracts and processes text from the provided URLs and creates an index.

        This endpoint handles the extraction of text from given URLs, saving the extracted text
        to a temporary file, and generating an index for the extracted text. The index can then
        be used for querying within the text. The process involves several steps
        including text extraction, file saving, and index generation.

        Args:
            data (ProcessUrlsInput): An object containing the list of URLs to be processed, along with
                                     additional data like chunk size, chunk overlap, embedding model,
                                     and a flag to determine if the extracted texts should be saved
                                     to a single file or multiple files.
            api_key (str): APIKeyHeader
        Returns:
            ProcessUrlsOutput: An object containing the name of the generated index and metadata
                               about the indexing process. This metadata includes details like the
                               URLs processed, text extraction settings, and the configuration used
                               for index generation.
        """
        # Initialize Process URLs Util Class
        process_urls_util = ProcessURLsUtil(
            indices_dir=self.indices_dir, document_dir=self.document_dir, reddit_util=self.reddit_util
        )

        # (1) Process URLs content and extract text
        start1 = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /process/urls")
        urls_meta = []
        for url in data.urls:
            urls_meta.append({
                'url': url,
                'description': data.url_description
            })
        extracted_texts = await process_urls_util.extract_text(urls_meta)
        end = time.time()
        self.logger.info(f"Text Extracted from `{len(data.urls)}` URLS in {round((end - start1) * 1000, 2)} ms")

        # (2) Write extracted text to tmp file
        start = time.time()
        tmp_document_save_path = write_data_to_file(
            document_dir=self.document_dir,
            full_documents=extracted_texts,
            single_file_flag=data.save_to_one_file,
        )
        end = time.time()
        self.logger.info(f"Extracted text saved to `{tmp_document_save_path}` in {round((end - start) * 1000, 2)} ms")

        # (3) Generate FAISS index
        start = time.time()
        document_directory = os.path.join(self.document_dir, tmp_document_save_path)
        index_directory = os.path.join(self.indices_dir, tmp_document_save_path)
        index_data(
            document_directory=document_directory,
            index_directory=index_directory,
            chunk_size=data.chunk_size,
            chunk_overlap=data.chunk_overlap,
            embedding_model=data.embedding_model
        )
        result = {"index_name": os.path.basename(index_directory)}
        end = time.time()
        self.logger.info(f"FAISS index saved to `{index_directory}` in {round((end - start) * 1000, 2)} ms")

        # (4) Construct a metadata dictionary from the processing data.
        result["index_meta"] = {
            "urls_meta": {
                'urls': data.urls, 'description': data.url_description
            },
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }
        end = time.time()
        self.logger.info(f"Finished /process/urls in {round((end - start1) * 1000, 2)} ms")
        return ProcessUrlsOutput.model_validate(result)

    @app.post(
        "/process/text",
        tags=["process"],
        summary="Process text and generate text index.",
        response_model=ProcessTextOutput,
    )
    async def process_text(
            self, data: ProcessTextInput, api_key: str = Depends(auth_util.get_api_key)
    ) -> ProcessTextOutput:
        """
        Processes a given text input, creates a searchable index, and returns metadata about the process.

        This endpoint is designed to handle the processing of raw text data. It involves several steps,
        including the extraction and processing of text, saving the processed text to a temporary file,
        and then generating a FAISS index for the text. The endpoint is useful for making the provided
        text searchable or queryable.

        Args:
            data (ProcessTextInput): An object containing the text to be processed, along with additional
                                     processing parameters. These parameters include a description of the
                                     text, chunk size for indexing, chunk overlap, the embedding model to
                                     be used, and a flag to determine if the processed text should be saved
                                     to a single file or multiple files.
            api_key (str): APIKeyHeader

        Returns:
            ProcessTextOutput: An object containing details about the generated index and the text processing.
                               This includes the name of the index, the processed text metadata (like a snippet
                               of the text and its description), and settings used for the indexing process.
        """
        # Initialize Process URLs Util Class
        process_text_util = ProcessTextUtil(indices_dir=self.indices_dir, document_dir=self.document_dir)

        # (1) Process URLs content and extract text
        start1 = time.time()
        self.logger.info(f"Request received api key: {api_key}. Endpoint: /process/text")
        text_meta = {
            'text': data.text, 'description': data.text_description
        }
        extracted_text = await process_text_util.process_text(text_meta)
        end = time.time()
        self.logger.info(f"Text Processed of len `{len(data.text)}` in {round((end - start1) * 1000, 2)} ms")

        # (2) Write extracted text to tmp file
        start = time.time()
        tmp_document_save_path = write_data_to_file(
            document_dir=self.document_dir,
            full_documents=extracted_text,
            single_file_flag=data.save_to_one_file,
        )
        end = time.time()
        self.logger.info(f"Extracted text saved to `{tmp_document_save_path}` in {round((end - start) * 1000, 2)} ms")

        # (3) Generate FAISS index
        start = time.time()
        document_directory = os.path.join(self.document_dir, tmp_document_save_path)
        index_directory = os.path.join(self.indices_dir, tmp_document_save_path)
        index_data(
            document_directory=document_directory,
            index_directory=index_directory,
            chunk_size=data.chunk_size,
            chunk_overlap=data.chunk_overlap,
            embedding_model=data.embedding_model
        )
        result = {"index_name": os.path.basename(index_directory)}
        end = time.time()
        self.logger.info(f"FAISS index saved to `{index_directory}` in {round((end - start) * 1000, 2)} ms")

        # (4) Construct a metadata dictionary from the processing data.
        result["index_meta"] = {
            "text_meta": {
                'text_snippet': data.text[:100] + '...', 'description': data.text_description
            },
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }
        end = time.time()
        self.logger.info(f"Finished /process/urls in {round((end - start1) * 1000, 2)} ms")
        return ProcessTextOutput.model_validate(result)

    @app.get(
        "/health",
        tags=["healthcheck"],
        summary="Perform a Health Check",
        response_description="Return HTTP Status Code 200 (OK)",
        status_code=status.HTTP_200_OK,
        response_model=HealthCheck,
    )
    def health_check(self) -> HealthCheck:
        """
        Health check endpoint to ensure the server is up and running.

        This endpoint performs basic checks to confirm that the FastAPI server and
        Ray Serve are operational. It can be extended to include more comprehensive
        checks depending on the application's requirements.

        Returns:

            HealthCheck: Response model to validate and return when performing a health check.
        """
        return HealthCheck(status="OK")


def main():
    """
    How to start and stop ray and server?
    >> ray start --head
    >> python server.py -vvv
    >> python -c "from ray import serve; serve.shutdown()"
    >> ray stop
    """
    _cpus = multiprocessing.cpu_count()
    _num_cpus = _cpus // 2
    parser = argparse.ArgumentParser(description='Start VerbalVista Ray Server!')
    parser.add_argument('--num_replicas', type=int, default=1, help='Number of replicas.')
    parser.add_argument('--num_cpus', type=int, default=_num_cpus, help='Number of CPUs.')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs.')
    parser.add_argument('--max_concurrent_queries', type=int, default=100, help='Max concurrent queries.')
    parser.add_argument('--health_check_period_s', type=int, default=10, help='Health check period (seconds).')
    parser.add_argument('--health_check_timeout_s', type=int, default=30, help='Health check timeout (seconds).')
    parser.add_argument('--graceful_shutdown_timeout_s', type=int, default=20, help='Graceful shutdown timeout (seconds).')
    parser.add_argument('--graceful_shutdown_wait_loop_s', type=int, default=2, help='Graceful shutdown wait loop (seconds).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address.')
    parser.add_argument('--port', type=int, default=8000, help='Port number.')
    parser.add_argument('--route_prefix', type=str, default='/', help='Route prefix.')
    parser.add_argument('--server_name', type=str, default='verbal_vista', help='Server name.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (e.g., -v, -vv)')

    args = parser.parse_args()

    # Set logging level based on verbosity count
    if args.verbose >= 2:
        logging_level = logging.DEBUG
    elif args.verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    deployment = VerbalVistaAssistantDeployment.options(
        name="VerbalVistaServer",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
        max_concurrent_queries=args.max_concurrent_queries,
        health_check_period_s=args.health_check_period_s,
        health_check_timeout_s=args.health_check_timeout_s,
        graceful_shutdown_timeout_s=args.graceful_shutdown_timeout_s,
        graceful_shutdown_wait_loop_s=args.graceful_shutdown_wait_loop_s
    ).bind(logging_level=logging_level)

    serve.run(
        deployment,
        host=args.host,
        port=args.port,
        route_prefix=args.route_prefix,
        name=args.server_name
    )


if __name__ == "__main__":
    main()
