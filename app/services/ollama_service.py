import httpx

from app.config import get_settings


class OllamaService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _url(self, path: str) -> str:
        return f"{self.settings.ollama_base_url.rstrip('/')}{path}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": self.settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.settings.chat_keep_alive,
            "options": {
                "temperature": temperature,
            },
        }

        with httpx.Client(timeout=180.0) as client:
            response = client.post(self._url("/api/chat"), json=payload)
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")

        if not isinstance(content, str):
            raise RuntimeError("Ollama chat returned invalid content")

        return content.strip()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.settings.ollama_embed_model,
            "input": texts,
            "keep_alive": self.settings.embed_keep_alive,
        }

        with httpx.Client(timeout=180.0) as client:
            response = client.post(self._url("/api/embed"), json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embed returned invalid embeddings payload")

        return embeddings

    def tags(self) -> dict:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(self._url("/api/tags"))
            response.raise_for_status()
            return response.json()