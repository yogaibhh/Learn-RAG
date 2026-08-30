"""Embedding lokal — alternatif offline kalau tidak mau pakai API sama sekali.

Memakai bobot terbuka `jina-embeddings-v5-text-nano` (239 juta parameter) lewat
sentence-transformers. Sengaja dipilih varian *nano*, bukan *small* (677 juta),
supaya masih nyaman jalan di CPU laptop.

Antarmukanya sengaja dibuat sama persis dengan `JinaEmbedder`, jadi index dan
retrieval tidak perlu tahu embedding-nya datang dari mana.

Dua hal yang perlu disadari sebelum memakai jalur ini:

1. **Lisensi.** Bobot Jina v5 berlisensi CC BY-NC 4.0 — non-komersial. Untuk
   belajar seperti project ini aman; untuk produk komersial tidak.
2. **`trust_remote_code=True`.** Model ini menjalankan kode Python dari repo
   Hugging Face-nya di mesin kamu. Itu memang wajib untuk arsitektur ini, tapi
   artinya kamu sedang mempercayai penerbit model tersebut.

Late chunking tidak tersedia di jalur ini; itu fitur sisi server milik API.

Pasang dulu:  uv sync --extra local
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rag.config import LOCAL_MODEL
from rag.embed import EmbeddingError, normalize

_INSTALL_HINT = (
    "Embedding lokal butuh dependency tambahan yang belum terpasang.\n"
    "Jalankan:  uv sync --extra local\n"
    "(unduhannya besar, sekitar 2,5 GB untuk torch dan kawan-kawan)"
)


class LocalEmbedder:
    """Pembungkus sentence-transformers dengan antarmuka sama seperti JinaEmbedder."""

    name = LOCAL_MODEL
    supports_late_chunking = False

    def __init__(self, model_name: str = LOCAL_MODEL, batch_size: int = 8):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - bergantung pada extra opsional
            raise EmbeddingError(_INSTALL_HINT) from exc

        # trust_remote_code wajib: arsitektur Jina v5 didefinisikan di repo modelnya.
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self.batch_size = batch_size
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def close(self) -> None:
        """Ada supaya bisa dipakai bergantian dengan JinaEmbedder."""

    def __enter__(self) -> "LocalEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _encode(self, texts: Sequence[str], task: str) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            task=task,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_passages(self, texts: Sequence[str], late_chunking: bool = False) -> np.ndarray:
        # Argumen late_chunking diterima tapi diabaikan: itu fitur sisi server.
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], task="retrieval.query")[0]
