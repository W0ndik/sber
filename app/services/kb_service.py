import re
from datetime import datetime, timezone
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.config import get_settings
from app.db import AppDB
from app.services.ollama_service import OllamaService
from app.services.vector_store import VectorStore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class KnowledgeBaseService:
    def __init__(
        self,
        db: AppDB,
        ollama: OllamaService,
        vector_store: VectorStore,
    ) -> None:
        self.settings = get_settings()
        self.db = db
        self.ollama = ollama
        self.vector_store = vector_store
        self.base_dir = Path(self.settings.knowledge_base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _iter_files(self) -> list[Path]:
        files = [
            path
            for path in self.base_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        files.sort()
        return files

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)

        if suffix == ".docx":
            doc = DocxDocument(str(path))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)

        return ""

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)

            if end < len(text):
                paragraph_break = text.rfind("\n\n", start, end)
                sentence_break = text.rfind(". ", start, end)
                natural_break = max(paragraph_break, sentence_break)

                if natural_break > start + chunk_size // 2:
                    end = natural_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(0, end - overlap)

        return chunks

    def _make_safe_filename(self, filename: str) -> str:
        original_name = Path(filename).name.strip()
        if not original_name:
            raise ValueError("Empty filename")

        suffix = Path(original_name).suffix.lower()
        stem = Path(original_name).stem

        stem = re.sub(r"[^\w\- .А-Яа-яЁё]+", "_", stem, flags=re.UNICODE).strip(" ._")
        if not stem:
            stem = "document"

        candidate = f"{stem}{suffix}"
        target = self.base_dir / candidate
        counter = 1

        while target.exists():
            candidate = f"{stem}_{counter}{suffix}"
            target = self.base_dir / candidate
            counter += 1

        return candidate

    def list_documents(self) -> list[dict]:
        result: list[dict] = []

        for path in self._iter_files():
            stat = path.stat()
            modified_at = datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(timespec="seconds")

            result.append(
                {
                    "filename": str(path.relative_to(self.base_dir)),
                    "size_bytes": int(stat.st_size),
                    "modified_at": modified_at,
                }
            )

        return result

    def save_documents(self, files: list[tuple[str, bytes]]) -> tuple[list[str], list[str]]:
        saved_files: list[str] = []
        skipped_files: list[str] = []

        for original_name, content in files:
            name = Path(original_name).name.strip()

            if not name:
                skipped_files.append(original_name or "<empty>")
                continue

            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped_files.append(name)
                continue

            if not content:
                skipped_files.append(name)
                continue

            safe_name = self._make_safe_filename(name)
            target = self.base_dir / safe_name
            target.write_bytes(content)
            saved_files.append(safe_name)

        return saved_files, skipped_files

    def delete_document(self, filename: str) -> bool:
        safe_name = Path(filename).name

        if safe_name != filename:
            return False

        target = self.base_dir / safe_name

        if not target.exists():
            return False

        if not target.is_file():
            return False

        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        target.unlink()
        return True

    def reindex(self) -> tuple[int, int]:
        files = self._iter_files()

        all_chunks: list[dict] = []
        indexed_docs: list[tuple[str, int]] = []

        for path in files:
            text = self._extract_text(path)
            chunks = self._chunk_text(text)

            if not chunks:
                continue

            relative_source = str(path.relative_to(self.base_dir))
            indexed_docs.append((relative_source, len(chunks)))

            for chunk_index, chunk_text in enumerate(chunks):
                all_chunks.append(
                    {
                        "id": len(all_chunks) + 1,
                        "source": relative_source,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )

        self.db.replace_indexed_documents(indexed_docs)

        if not all_chunks:
            return 0, 0

        embeddings = self.ollama.embed_texts([item["text"] for item in all_chunks])

        self.vector_store.reset_collection(vector_size=len(embeddings[0]))
        self.vector_store.upsert_chunks(all_chunks, embeddings)

        return len(indexed_docs), len(all_chunks)

    def search(self, query: str, limit: int) -> list[dict]:
        query_vector = self.ollama.embed_texts([query])[0]
        hits = self.vector_store.search(query_vector=query_vector, limit=limit)

        results: list[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "source": str(payload.get("source", "")),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "text": str(payload.get("text", "")),
                    "score": float(hit.score),
                }
            )

        return results