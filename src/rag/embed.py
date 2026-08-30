"""Tahap 3 — mengubah teks jadi vektor lewat Jina Embeddings v5.

Dua parameter Jina yang jadi inti kualitas retrieval di sini:

`task`
    Embedding di sini **asimetris**. Chunk di-embed dengan `retrieval.passage`,
    pertanyaan dengan `retrieval.query`. Model memakai adapter berbeda untuk
    keduanya karena pertanyaan dan jawaban memang tidak berbentuk sama:
    "Apa itu overfitting?" tidak mirip secara permukaan dengan paragraf yang
    menjelaskan overfitting. Memakai task yang sama untuk keduanya adalah
    kesalahan klasik yang diam-diam menurunkan kualitas pencarian.

`late_chunking`
    Biasanya tiap chunk di-embed sendiri-sendiri, jadi chunk yang berbunyi
    "Model ini dilatih dengan ..." kehilangan acuan "model ini". Dengan late
    chunking, semua chunk satu dokumen dikirim dalam satu request, model
    membaca semuanya sebagai satu kesatuan, lalu barulah embedding tiap chunk
    diambil. Hasilnya tiap vektor sudah mengandung konteks dokumen penuh.

Konsekuensi late chunking: satu request = satu dokumen (isinya digabung di
server), jadi batching-nya per-dokumen, bukan per-N-chunk sembarangan.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import httpx
import numpy as np

from rag.config import JINA_API_URL, JINA_DIMENSIONS, JINA_MODEL, jina_api_key

# Free tier Jina: 100 request/menit, 100.000 token/menit.
# Diberi margin supaya tidak menyenggol batas gara-gara selisih hitungan token.
FREE_TIER_TOKENS_PER_MINUTE = 90_000

# Perkiraan kasar untuk teks Indonesia. Dipakai hanya untuk mengatur laju,
# bukan untuk penagihan, jadi tidak perlu presisi.
CHARS_PER_TOKEN = 3.5

# Batas aman satu request late-chunking. Model menampung 32K token; angka ini
# kira-kira 17K token, cukup longgar untuk artikel Wikipedia terpanjang.
MAX_CHARS_PER_REQUEST = 60_000


class EmbeddingError(RuntimeError):
    pass


class RateLimiter:
    """Pembatas laju sederhana berbasis jendela satu menit bergulir."""

    def __init__(self, tokens_per_minute: int):
        self.tokens_per_minute = tokens_per_minute
        self._events: list[tuple[float, int]] = []

    def wait_for(self, tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._events = [(t, n) for t, n in self._events if now - t < 60.0]
            used = sum(n for _, n in self._events)

            if used + tokens <= self.tokens_per_minute or not self._events:
                self._events.append((now, tokens))
                return

            oldest = min(t for t, _ in self._events)
            sleep_for = 60.0 - (now - oldest) + 0.5
            print(f"    (batas laju tercapai: tunggu {sleep_for:.0f} detik)")
            time.sleep(sleep_for)


def estimate_tokens(texts: Sequence[str]) -> int:
    return max(1, int(sum(len(t) for t in texts) / CHARS_PER_TOKEN))


def group_by_size(
    texts: Sequence[str], max_chars: int = MAX_CHARS_PER_REQUEST
) -> list[list[int]]:
    """Membagi indeks teks jadi kelompok yang muat dalam satu request.

    Mengembalikan indeks, bukan teksnya, supaya pemanggil bisa menyusun hasil
    kembali ke urutan semula.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    size = 0

    for index, text in enumerate(texts):
        if current and size + len(text) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(index)
        size += len(text)

    if current:
        groups.append(current)
    return groups


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Menormalkan tiap baris jadi panjang 1.

    Setelah ini cosine similarity cukup dihitung sebagai perkalian titik biasa,
    yang jauh lebih murah daripada membagi norma berulang kali saat pencarian.
    """
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix / norm if norm else matrix

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class JinaEmbedder:
    """Klien Jina Embeddings v5 dengan pembatas laju dan percobaan ulang."""

    name = JINA_MODEL
    supports_late_chunking = True

    def __init__(self, api_key: str | None = None, dimensions: int = JINA_DIMENSIONS):
        key = api_key or jina_api_key()
        if not key:
            raise EmbeddingError(
                "JINA_API_KEY belum diisi.\n"
                "Ambil key gratis (tanpa kartu kredit) di https://jina.ai/?sui=apikey\n"
                "lalu isikan ke berkas .env. Alternatif tanpa key sama sekali:\n"
                "  uv run python cli.py index --embedder bm25    (BM25 saja, tanpa vektor)\n"
                "  uv sync --extra local && uv run python cli.py index --embedder local"
            )
        self._key = key
        self.dimensions = dimensions
        self._limiter = RateLimiter(FREE_TIER_TOKENS_PER_MINUTE)
        self._client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=15.0),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JinaEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, payload: dict, max_attempts: int = 5) -> list[list[float]]:
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.post(JINA_API_URL, json=payload)
            except httpx.RequestError as exc:
                if attempt == max_attempts:
                    raise EmbeddingError(f"Gagal menghubungi Jina: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 200:
                data = response.json()["data"]
                # API tidak menjamin urutan balasan, jadi diurutkan lewat "index".
                return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == max_attempts:
                    raise EmbeddingError(
                        f"Jina membalas {response.status_code} setelah {max_attempts} "
                        f"percobaan: {response.text[:300]}"
                    )
                retry_after = response.headers.get("retry-after")
                sleep_for = float(retry_after) if retry_after else delay
                print(f"    (HTTP {response.status_code}: coba lagi dalam {sleep_for:.0f} detik)")
                time.sleep(sleep_for)
                delay *= 2
                continue

            if response.status_code in (401, 403):
                raise EmbeddingError(
                    f"Jina menolak API key (HTTP {response.status_code}). "
                    "Periksa JINA_API_KEY di berkas .env."
                )

            raise EmbeddingError(
                f"Jina membalas HTTP {response.status_code}: {response.text[:300]}"
            )

        raise EmbeddingError("Percobaan habis tanpa hasil.")

    def _embed(self, texts: Sequence[str], task: str, late_chunking: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        self._limiter.wait_for(estimate_tokens(texts))
        vectors = self._post(
            {
                "model": JINA_MODEL,
                "task": task,
                "dimensions": self.dimensions,
                "late_chunking": late_chunking,
                "input": list(texts),
            }
        )
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_passages(self, texts: Sequence[str], late_chunking: bool = True) -> np.ndarray:
        """Meng-embed teks yang akan disimpan di index.

        Semua teks di satu panggilan diasumsikan berasal dari dokumen yang sama.
        Itu syarat late chunking, karena server menggabungkan seluruh input jadi
        satu badan teks sebelum memotongnya kembali.
        """
        return self._embed(texts, task="retrieval.passage", late_chunking=late_chunking)

    def embed_query(self, text: str) -> np.ndarray:
        """Meng-embed satu pertanyaan. Late chunking tidak berlaku untuk query."""
        return self._embed([text], task="retrieval.query", late_chunking=False)[0]


def embed_documents(
    embedder: JinaEmbedder,
    groups: Iterable[tuple[str, list[str]]],
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Meng-embed chunk per dokumen supaya late chunking bekerja.

    `groups` berisi pasangan (doc_id, daftar teks chunk milik dokumen itu).
    Dokumen yang terlalu panjang untuk satu request dipecah lagi, tapi tetap
    tidak pernah dicampur dengan dokumen lain.
    """
    result: dict[str, np.ndarray] = {}
    groups = list(groups)

    for number, (doc_id, texts) in enumerate(groups, start=1):
        parts = [
            embedder.embed_passages([texts[i] for i in indices])
            for indices in group_by_size(texts)
        ]
        result[doc_id] = np.vstack(parts)
        if verbose:
            print(f"  [{number:2d}/{len(groups)}] {doc_id}: {len(texts)} chunk")

    return result
