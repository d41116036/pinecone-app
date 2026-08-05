import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


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

DEFAULT_GEMINI_APP_BASE_URL = "http://54.204.110.222/gemini"


class GeminiConfigurationError(Exception):
    """Raised when gemini-app client configuration is invalid."""


class GeminiServiceError(Exception):
    """Raised when gemini-app API calls fail."""


def _base_url() -> str:
    return os.getenv("GEMINI_APP_BASE_URL", DEFAULT_GEMINI_APP_BASE_URL).rstrip("/")


def _post_json(path: str, payload: dict) -> dict:
    url = f"{_base_url()}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info("Calling gemini-app: url=%s payload_keys=%s", url, list(payload.keys()))
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code >= 500 and "not set" in error_body.lower():
            raise GeminiConfigurationError(error_body) from exc
        raise GeminiServiceError(
            f"gemini-app request failed ({exc.code}) at {url}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GeminiServiceError(f"gemini-app unreachable at {url}: {exc}") from exc


def create_embedding(text: str) -> list[float]:
    body = _post_json("/gemini/embedding", {"text": text})
    embedding = body.get("embedding") or []
    if not embedding:
        raise GeminiServiceError("gemini-app returned an empty embedding.")
    return embedding
