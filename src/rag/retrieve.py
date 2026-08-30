"""Tahap 5 — mencari chunk yang relevan.

Ada tiga mode, dan perbedaannya justru inti pelajaran di project ini.

**Dense** membandingkan makna. Query di-embed jadi vektor lalu dibandingkan
dengan vektor tiap chunk. Kuat untuk parafrase dan sinonim: "cara komputer
belajar sendiri" bisa menemukan paragraf tentang pembelajaran mesin walau tidak
satu kata pun sama.

**BM25** membandingkan kata. Metode statistik klasik yang memberi bobot besar ke
kata langka dan menghukum dokumen kepanjangan. Kuat untuk nama diri, singkatan,
dan istilah teknis: cari "AlphaGo" dan BM25 langsung menemukannya, sementara
dense bisa saja mengembalikan paragraf umum tentang permainan papan.

**Hybrid** menggabungkan keduanya dengan Reciprocal Rank Fusion. RRF hanya
memakai *peringkat*, bukan skor mentah, karena skor cosine (0..1) dan skor BM25
(tak terbatas ke atas) tidak sebanding dan tidak bisa dijumlahkan begitu saja.
Rumusnya: skor = jumlah dari 1 / (k + peringkat) untuk tiap metode. Konstanta k
meredam pengaruh peringkat teratas supaya satu metode tidak mendominasi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunk import Chunk
from rag.index import Index

# Konstanta standar dari makalah RRF. Makin besar, makin kecil bedanya antara
# peringkat 1 dan peringkat 10, sehingga suara kedua metode makin seimbang.
RRF_K = 60

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Kata yang muncul di hampir semua dokumen tidak membantu membedakan apa pun,
# dan justru bikin BM25 mengembalikan chunk yang panjang tapi tidak relevan.
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah",
    "ini", "itu", "atau", "dalam", "tidak", "akan", "juga", "oleh", "sebagai",
    "dapat", "telah", "bisa", "ada", "lebih", "karena", "para", "suatu",
    "sebuah", "banyak", "seperti", "antara", "namun", "tersebut", "yaitu",
    "apa", "bagaimana", "mengapa", "kapan", "siapa", "mana", "saja",
}


def tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana untuk BM25.

    Bahasa Indonesia tidak butuh stemming seagresif bahasa Inggris untuk kasus
    ini, jadi cukup huruf kecil, ambil kata, buang stopword.
    """
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass
class Hit:
    """Satu hasil pencarian beserta alasan kenapa ia muncul."""

    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    bm25_rank: int | None = None

    def why(self) -> str:
        """Penjelasan singkat asal-usul peringkat ini."""
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense #{self.dense_rank}")
        if self.bm25_rank is not None:
            parts.append(f"bm25 #{self.bm25_rank}")
        return " + ".join(parts) if parts else "—"


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    """Mengambil indeks k skor tertinggi, terurut menurun.

    argpartition dipakai supaya tidak perlu mengurutkan seluruh larik hanya
    untuk mengambil sepuluh teratas.
    """
    k = min(k, len(scores))
    if k <= 0:
        return []
    partial = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in partial[np.argsort(-scores[partial])]]


class Retriever:
    """Pencari chunk di atas sebuah index.

    `embedder` boleh None. Tanpa embedder, mode dense dan hybrid otomatis
    turun jadi BM25 saja, bukan gagal.
    """

    def __init__(self, index: Index, embedder=None):
        self.index = index
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(chunk.embed_text()) for chunk in index.chunks])

    @property
    def can_dense(self) -> bool:
        return self.index.has_vectors and self.embedder is not None

    def dense(self, query: str, k: int = 5) -> list[Hit]:
        """Pencarian berdasarkan kemiripan makna."""
        if not self.can_dense:
            raise RuntimeError(
                "Mode dense butuh index bervektor dan sebuah embedder. "
                "Bangun index dengan embedder, atau pakai mode bm25."
            )

        query_vector = self.embedder.embed_query(query)
        # Vektor sudah dinormalkan sejak dibuat, jadi perkalian titik = cosine.
        scores = self.index.embeddings @ query_vector

        return [
            Hit(chunk=self.index.chunks[i], score=float(scores[i]), rank=rank, dense_rank=rank)
            for rank, i in enumerate(_top_indices(scores, k), start=1)
        ]

    def bm25(self, query: str, k: int = 5) -> list[Hit]:
        """Pencarian berdasarkan kecocokan kata."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        return [
            Hit(chunk=self.index.chunks[i], score=float(scores[i]), rank=rank, bm25_rank=rank)
            for rank, i in enumerate(_top_indices(scores, k), start=1)
        ]

    def hybrid(self, query: str, k: int = 5, pool: int = 30) -> list[Hit]:
        """Menggabungkan dense dan BM25 dengan Reciprocal Rank Fusion.

        Tiap metode diminta `pool` kandidat -- lebih banyak dari k -- supaya
        chunk yang cuma kuat di salah satu metode masih punya kesempatan naik
        setelah digabung.
        """
        if not self.can_dense:
            return self.bm25(query, k)

        dense_hits = self.dense(query, pool)
        bm25_hits = self.bm25(query, pool)

        fused: dict[str, dict] = {}
        for hits, key in ((dense_hits, "dense_rank"), (bm25_hits, "bm25_rank")):
            for hit in hits:
                entry = fused.setdefault(
                    hit.chunk.id,
                    {"chunk": hit.chunk, "score": 0.0, "dense_rank": None, "bm25_rank": None},
                )
                entry["score"] += 1.0 / (RRF_K + hit.rank)
                entry[key] = hit.rank

        ranked = sorted(fused.values(), key=lambda entry: -entry["score"])[:k]
        return [
            Hit(
                chunk=entry["chunk"],
                score=entry["score"],
                rank=rank,
                dense_rank=entry["dense_rank"],
                bm25_rank=entry["bm25_rank"],
            )
            for rank, entry in enumerate(ranked, start=1)
        ]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode == "dense":
            return self.dense(query, k)
        if mode == "bm25":
            return self.bm25(query, k)
        if mode == "hybrid":
            return self.hybrid(query, k)
        raise ValueError(f"Mode tidak dikenal: {mode!r}. Pilih dense, bm25, atau hybrid.")
