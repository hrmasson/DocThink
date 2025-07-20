import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("doc_think.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from presentation.api.rag_router import router as rag_router

logger = logging.getLogger(__name__)
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
