"""Tahap 4 — membangun dan memuat index.

Index di sini cuma tiga berkas di data/index/:

    chunks.jsonl     teks tiap chunk beserta asal-usulnya
    embeddings.npy   matriks (jumlah_chunk x dimensi), sudah dinormalkan
    meta.json        catatan cara index ini dibuat

Kenapa numpy dan bukan FAISS atau vector database? Korpus ini 474 chunk.
Perkalian matriks 474 x 1024 selesai dalam hitungan mikrodetik, jadi struktur
data pencarian aproksimasi belum ada gunanya -- yang ada malah menyembunyikan
apa yang sebenarnya terjadi. Baru kalau korpusnya jutaan chunk, barulah index
seperti HNSW jadi masuk akal.

Karena embeddings sudah dinormalkan sejak dibuat, cosine similarity di tahap
pencarian tinggal satu perkalian titik.

embeddings.npy sengaja tidak di-commit: itu hasil turunan yang bisa dibangun
ulang kapan saja dari data/raw/. Yang di-commit cuma korpus mentahnya.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.chunk import Chunk, chunk_corpus
from rag.config import CHUNKS_PATH, EMBEDDINGS_PATH, INDEX_DIR, INDEX_META_PATH


@dataclass
class Index:
    """Index yang sudah dimuat ke memori."""

    chunks: list[Chunk]
    embeddings: np.ndarray | None
    meta: dict

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def has_vectors(self) -> bool:
        return self.embeddings is not None and len(self.embeddings) > 0

    @property
    def embedder_name(self) -> str:
        return self.meta.get("embedder", "tidak diketahui")

    def describe(self) -> str:
        docs = len({chunk.doc_id for chunk in self.chunks})
        if self.has_vectors:
            vectors = f"{self.embeddings.shape[0]} vektor x {self.embeddings.shape[1]} dimensi"
        else:
            vectors = "tanpa vektor (mode BM25 saja)"
        return f"{len(self.chunks)} chunk dari {docs} dokumen, {vectors}"


def group_chunks_by_document(chunks: list[Chunk]) -> list[tuple[str, list[int]]]:
    """Mengelompokkan indeks chunk menurut dokumen asalnya.

    Late chunking mensyaratkan satu request berisi satu dokumen, jadi urutan
    chunk per dokumen harus dijaga utuh di sini.
    """
    groups: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        groups.setdefault(chunk.doc_id, []).append(index)
    return list(groups.items())


def build(embedder=None, verbose: bool = True) -> Index:
    """Membangun index dari data/raw/ lalu menyimpannya ke disk.

    `embedder` boleh None. Kalau None, index dibuat tanpa vektor dan pencarian
    hanya bisa memakai BM25. Itu berguna untuk mencoba sistem ini tanpa API key
    sama sekali.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = chunk_corpus()
    if verbose:
        docs = len({chunk.doc_id for chunk in chunks})
        print(f"{len(chunks)} chunk dari {docs} dokumen.")

    embeddings: np.ndarray | None = None

    if embedder is not None:
        if verbose:
            print(f"Meng-embed dengan {embedder.name} ...")

        groups = group_chunks_by_document(chunks)
        matrix = np.zeros((len(chunks), embedder.dimensions), dtype=np.float32)

        for number, (doc_id, indices) in enumerate(groups, start=1):
            texts = [chunks[i].embed_text() for i in indices]
            vectors = embedder.embed_passages(texts)
            matrix[indices] = vectors
            if verbose:
                print(f"  [{number:2d}/{len(groups)}] {chunks[indices[0]].title} — {len(indices)} chunk")

        embeddings = matrix
        np.save(EMBEDDINGS_PATH, embeddings)
    elif EMBEDDINGS_PATH.exists():
        # Vektor lama tidak lagi sejalan dengan chunk yang baru dibangun.
        EMBEDDINGS_PATH.unlink()

    with CHUNKS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.to_json() + "\n")

    meta = {
        "dibuat_pada": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jumlah_chunk": len(chunks),
        "jumlah_dokumen": len({chunk.doc_id for chunk in chunks}),
        "embedder": getattr(embedder, "name", None),
        "dimensi": int(embeddings.shape[1]) if embeddings is not None else None,
        "late_chunking": bool(getattr(embedder, "supports_late_chunking", False)),
    }
    INDEX_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    index = Index(chunks=chunks, embeddings=embeddings, meta=meta)
    if verbose:
        print(f"Index tersimpan di {INDEX_DIR}: {index.describe()}")
    return index


def load(index_dir: Path = INDEX_DIR) -> Index:
    """Memuat index dari disk."""
    chunks_path = index_dir / CHUNKS_PATH.name
    if not chunks_path.exists():
        raise SystemExit(
            f"Index belum ada di {index_dir}.\n"
            "Bangun dulu:  uv run python cli.py index"
        )

    chunks = [
        Chunk.from_dict(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    embeddings_path = index_dir / EMBEDDINGS_PATH.name
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None

    meta_path = index_dir / INDEX_META_PATH.name
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if embeddings is not None and len(embeddings) != len(chunks):
        raise SystemExit(
            f"Index tidak konsisten: {len(chunks)} chunk tapi {len(embeddings)} vektor.\n"
            "Bangun ulang:  uv run python cli.py index"
        )

    return Index(chunks=chunks, embeddings=embeddings, meta=meta)
