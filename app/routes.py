import logging
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pypdf import PdfReader

from app.gemini_client import (
    GeminiConfigurationError,
    GeminiServiceError,
    create_embedding,
)
from app.models import (
    PdfUploadResponse,
    RetrieveDocMatch,
    RetrieveDocRequest,
    RetrieveDocResponse,
    RetrieveNamespaceResponse,
)
from app.pinecone_client import list_namespaces, query_vectors, upsert_vectors
from app.text_utils import chunk_text

router = APIRouter(prefix="/pinecone")
logger = logging.getLogger(__name__)


@router.post("/pdf", response_model=PdfUploadResponse)
def upload_pdf(
    file: UploadFile = File(...),
    namespace: str = Form(...),
) -> PdfUploadResponse:
    filename = file.filename or "upload.pdf"
    namespace = namespace.strip()
    if not namespace:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="namespace must not be blank.",
        )
    content_type = (file.content_type or "").lower()
    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

    if not is_pdf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    data = file.file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF is empty.",
        )

    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read PDF: {exc}",
        ) from exc

    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF has no extractable text.",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF text could not be split into chunks.",
        )

    vectors: list[dict] = []
    embeddings: list[list[float]] = []
    try:
        for index, chunk in enumerate(chunks):
            embedding = create_embedding(chunk)
            embeddings.append(embedding)
            vectors.append(
                {
                    "id": f"{filename}-{index}",
                    "values": embedding,
                    "metadata": {
                        "filename": filename,
                        "namespace": namespace,
                        "chunk_index": index,
                        "page_count": len(reader.pages),
                        "char_count": len(chunk),
                        "text": chunk,
                    },
                }
            )
    except GeminiConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GeminiServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    try:
        upsert_vectors(vectors, namespace=namespace)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upsert embedding to Pinecone: {exc}",
        ) from exc

    return PdfUploadResponse(
        filename=filename,
        page_count=len(reader.pages),
        text=text,
        char_count=len(text),
        embedding=embeddings[0],
        namespace=namespace,
        chunk_count=len(vectors),
    )


@router.post("/retrievedoc", response_model=RetrieveDocResponse)
def retrieve_doc(payload: RetrieveDocRequest) -> RetrieveDocResponse:
    try:
        question_embedding = create_embedding(payload.question)
        matches = query_vectors(
            question_embedding,
            namespace=payload.namespace,
            top_k=payload.top_k,
        )
    except GeminiConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GeminiServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve documents from Pinecone: {exc}",
        ) from exc

    doc_matches: list[RetrieveDocMatch] = []
    texts: list[str] = []
    for match in matches:
        text = str(match.get("metadata", {}).get("text", "")).strip()
        if not text:
            continue
        texts.append(text)
        doc_matches.append(
            RetrieveDocMatch(
                id=match.get("id"),
                score=match.get("score"),
                text=text,
            )
        )

    logger.info(
        "retrievedoc namespace=%s top_k=%s returned=%s",
        payload.namespace,
        payload.top_k,
        len(texts),
    )
    return RetrieveDocResponse(
        texts=texts,
        matches=doc_matches,
        namespace=payload.namespace,
    )


@router.get("/retrievenamespace", response_model=RetrieveNamespaceResponse)
def retrieve_namespace() -> RetrieveNamespaceResponse:
    try:
        namespaces = list_namespaces()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to list Pinecone namespaces: {exc}",
        ) from exc
    return RetrieveNamespaceResponse(namespaces=namespaces)
