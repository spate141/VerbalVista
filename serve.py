import argparse
import structlog
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware

from utils.rag_utils.agent_util import QueryAgent
from utils.rag_utils.rag_util import load_index_and_metadata


app = FastAPI(
    title="Inference API for VerbalVista",
    description="🅛🅛🅜 + Your Data = 🩶",
    version="1.3",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[int] = 5
    max_lexical_retrieval_chunks: Optional[int] = 1


class Answer(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


@serve.deployment()
@serve.ingress(app)
class VerbalVistaAssistantDeployment:
    """
    Initialize VerbalVista Ray Assistant class with index_directory!
    """
    def __init__(self, index_directory=None):

        # setup logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        self.logger = structlog.get_logger()

        # get faiss index, lexical index and metadata for given index directory
        index_meta = load_index_and_metadata(index_directory=index_directory)
        faiss_index = index_meta["faiss_index"]
        lexical_index = index_meta["lexical_index"]
        metadata_dict = index_meta["metadata_dict"]

        # get query agent
        self.query_agent = QueryAgent(
            faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=None
        )

    def predict(self, query: Query, stream: bool = False) -> Dict[str, Any]:
        """
        Generate response for /predict endpoint.
        """
        result = self.query_agent(
            query=query.query,  temperature=query.temperature, embedding_model_name=query.embedding_model, stream=stream,
            llm_model=query.llm, num_chunks=query.max_semantic_retrieval_chunks, lexical_search_k=query.max_lexical_retrieval_chunks,
        )
        return result

    @app.post("/query")
    def query(self, query: Query) -> Answer:
        result = self.predict(query)
        return Answer.parse_obj(result)

    def check_health(self):
        if not self.query_agent.faiss_index:
            raise RuntimeError("uh-oh, server is broken.")


def main():
    """
    How to start and stop server?
    >> ray start --head
    >> python serve.py --index_dir=../data/indices/my_index/
    >> ray stop
    """
    parser = argparse.ArgumentParser(description='Start VerbalVista Ray Server!')
    parser.add_argument('--index_dir', type=str, required=True, help='Index directory.')
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

    args = parser.parse_args()

    deployment = VerbalVistaAssistantDeployment.options(
        name="VerbalVistaServer",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
        max_concurrent_queries=args.max_concurrent_queries,
        health_check_period_s=args.health_check_period_s,
        health_check_timeout_s=args.health_check_timeout_s,
        graceful_shutdown_timeout_s=args.graceful_shutdown_timeout_s,
        graceful_shutdown_wait_loop_s=args.graceful_shutdown_wait_loop_s
    ).bind(index_directory=args.index_dir)

    serve.run(
        deployment,
        host=args.host,
        port=args.port,
        route_prefix=args.route_prefix,
        name=args.server_name
    )


if __name__ == "__main__":
    main()
