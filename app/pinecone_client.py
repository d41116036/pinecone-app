import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app_env = os.getenv("APP_ENV", "dev").lower()
    env_file = project_root / f".env.{app_env}"
    load_dotenv(
        dotenv_path=env_file if env_file.exists() else project_root / ".env",
        override=True,
    )


_load_environment()
logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "chatbot"
DEFAULT_NAMESPACE = "example-namespace"


def _get_index(index_name: str = DEFAULT_INDEX_NAME):
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to .env.dev and save the file."
        )
    pc = Pinecone(api_key=pinecone_key)
    return pc.Index(index_name)


def upsert_vectors(
    vectors: list[dict[str, Any]],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    index_name: str = DEFAULT_INDEX_NAME,
):
    if not vectors:
        raise ValueError("vectors must be a non-empty list.")

    for i, vector in enumerate(vectors):
        if "id" not in vector or "values" not in vector:
            raise ValueError(f"vectors[{i}] must include 'id' and 'values' keys.")

    index = _get_index(index_name)
    result = index.upsert(vectors=vectors, namespace=namespace)
    logger.info(
        "Upserted %s vectors into index=%s namespace=%s",
        len(vectors),
        index_name,
        namespace,
    )
    return result


def list_namespaces(index_name: str = DEFAULT_INDEX_NAME) -> list[str]:
    index = _get_index(index_name)
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces") or {}
    names = sorted(namespaces.keys())
    logger.info("Found %s namespaces in index=%s", len(names), index_name)
    return names


def query_vectors(
    vector: list[float],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    top_k: int = 3,
    index_name: str = DEFAULT_INDEX_NAME,
) -> list[dict[str, Any]]:
    if not vector:
        raise ValueError("vector must be a non-empty list of floats.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    index = _get_index(index_name)
    response = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    matches: list[dict[str, Any]] = []
    for match in response.get("matches") or []:
        matches.append(
            {
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": match.get("metadata") or {},
            }
        )

    logger.info(
        "Query returned %s matches from index=%s namespace=%s",
        len(matches),
        index_name,
        namespace,
    )
    return matches
