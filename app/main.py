import logging

from fastapi import FastAPI

from app.routes import router as pinecone_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pinecone App", version="1.0.0")
app.include_router(pinecone_router)


@app.get("/pinecone/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
