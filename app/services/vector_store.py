from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        Path(self.settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.settings.qdrant_path)

    def reset_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.settings.qdrant_collection):
            self.client.delete_collection(self.settings.qdrant_collection)

        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        points: list[PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=embedding,
                    payload={
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            wait=True,
            points=points,
        )

    def search(self, query_vector: list[float], limit: int) -> list:
        if not self.client.collection_exists(self.settings.qdrant_collection):
            return []

        response = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points