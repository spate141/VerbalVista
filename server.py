import os
import time
import logging
import argparse
from ray import serve
from pydantic import BaseModel
from fastapi import FastAPI, status
from dotenv import load_dotenv; load_dotenv(".env")
from fastapi import UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

from utils.openai_utils import OpenAIWisperUtil
from utils.rag_utils.rag_util import index_data
from utils.data_parsing_utils import write_data_to_file
from utils.server_utils import (
    QueryUtil, ProcessTextUtil, ProcessURLsUtil, ProcessDocumentsUtil, ListIndicesUtil,
    QuestionInput, ProcessTextInput, ProcessUrlsInput, ProcessDocumentsInput,
    QuestionOutput, ProcessTextOutput, ProcessUrlsOutput, ProcessDocumentsOutput, ListIndicesOutput,
)
from utils.data_parsing_utils.reddit_comment_parser import RedditSubmissionCommentsFetcher


app = FastAPI(
    title="Inference API for VerbalVista",
    description="🅛🅛🅜 + Your Data = 🩶",
    version="1.6",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    @app.get(
        "/list/indices",
        tags=["list"],
        summary="Returns list of available indices.",
        response_model=ListIndicesOutput,
    )
    def get_indices(self) -> ListIndicesOutput:
        """
        Lists all the indices available in the system.

        Returns:

            ListIndicesOutput: Contains a list of all available indices.
        """
        start = time.time()
        list_indices_util = ListIndicesUtil()
        result = list_indices_util.list_indices(indices_dir=self.indices_dir)
        end = time.time()
        self.logger.info(f"Finished /list/indices in {round((end - start) * 1000, 2)} ms")
        return ListIndicesOutput.parse_list(result)

    @app.post(
        "/query",
        tags=["query"],
        summary="Use index and LLMs to generate response to user's question.",
        response_model=QuestionOutput,
    )
    def query(self, query: QuestionInput) -> QuestionOutput:
        """
        Processes a query and returns the predicted answer.

        Args:

            query (QuestionInput): The query input containing parameters for processing.

        Returns:

            QuestionOutput: The output containing the predicted answer.
        """
        start = time.time()
        query_util = QueryUtil(indices_dir=self.indices_dir, index_name=query.index_name)
        end = time.time()
        self.logger.info(f"Query agent initiated in {round((end - start) * 1000, 2)} ms")
        result = query_util.predict(
            query=query.query, temperature=query.temperature, embedding_model=query.embedding_model,
            llm_model=query.llm, max_semantic_retrieval_chunks=query.max_semantic_retrieval_chunks,
            max_lexical_retrieval_chunks=query.max_lexical_retrieval_chunks
        )
        end = time.time()
        self.logger.info(f"Finished /query in {round((end - start) * 1000, 2)} ms")
        return QuestionOutput.parse_obj(result)

    @app.post(
        "/process/documents",
        tags=["process"],
        summary="Process documents and generate document index.",
        response_model=ProcessDocumentsOutput,
    )
    async def process_documents(self, file: UploadFile = File(...), data: ProcessDocumentsInput = Depends(ProcessDocumentsInput.as_form)) -> ProcessDocumentsOutput:
        """
        Processes documents by indexing them and logs the operation.

        Args:

            file (UploadFile): The file to be processed.
            data (ProcessDocumentsInput): Additional data for processing the document.

        Returns:

            ProcessDocumentsOutput: The result of the document processing.
        """

        # Initialize Process Documents Util Class
        process_doc_util = ProcessDocumentsUtil(
            indices_dir=self.indices_dir, document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir,
            openai_wisper_util=self.openai_wisper_util
        )

        # (1) Read the file content
        start1 = time.time()
        file_meta = await process_doc_util.read_file(file, file_description=data.file_description)
        end = time.time()
        self.logger.info(f"Finished reading `{file_meta['name']}` in {round((end - start1) * 1000, 2)} ms")

        # (2) Process file content and extract text
        start = time.time()
        extracted_texts = process_doc_util.extract_text(file_meta=file_meta)
        end = time.time()
        self.logger.info(f"Text Extracted from `{file_meta['name']}` in {round((end - start) * 1000, 2)} ms")

        # (3) Write extracted text to tmp file
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
        _ = file_meta.pop("file")
        result["index_meta"] = {
            "file_meta": file_meta,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }
        end = time.time()
        self.logger.info(f"Finished /process/documents in {round((end - start1) * 1000, 2)} ms")
        return ProcessDocumentsOutput.parse_obj(result)

    @app.post(
        "/process/urls",
        tags=["process"],
        summary="Process URLs and generate URLs index.",
        response_model=ProcessUrlsOutput,
    )
    def process_urls(self, data: ProcessUrlsInput) -> ProcessUrlsOutput:
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
        urls_meta = []
        for url in data.urls:
            urls_meta.append({
                'url': url, 'description': data.url_description
            })
        extracted_texts = process_urls_util.extract_text(urls_meta)
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
        return ProcessUrlsOutput.parse_obj(result)

    @app.post(
        "/process/text",
        tags=["process"],
        summary="Process text and generate text index.",
        response_model=ProcessTextOutput,
    )
    def process_text(self, data: ProcessTextInput) -> ProcessTextOutput:
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

        Returns:

            ProcessTextOutput: An object containing details about the generated index and the text processing.
                               This includes the name of the index, the processed text metadata (like a snippet
                               of the text and its description), and settings used for the indexing process.
        """
        # Initialize Process URLs Util Class
        process_text_util = ProcessTextUtil(indices_dir=self.indices_dir, document_dir=self.document_dir)

        # (1) Process URLs content and extract text
        start1 = time.time()
        text_meta = {
            'text': data.text, 'description': data.text_description
        }
        extracted_text = process_text_util.process_text(text_meta)
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
        return ProcessTextOutput.parse_obj(result)

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
    How to start and stop server?
    >> ray start --head
    >> python server.py --index_dir=../data/indices/my_index/
    >> ray stop
    """
    parser = argparse.ArgumentParser(description='Start VerbalVista Ray Server!')
    parser.add_argument('--num_replicas', type=int, default=1, help='Number of replicas.')
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs.')
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
