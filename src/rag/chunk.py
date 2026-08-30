"""Tahap 2 — memecah dokumen jadi chunk.

Kenapa dokumen dipecah sama sekali? Dua alasan:

1. Presisi retrieval. Satu artikel Wikipedia bisa 60 ribu karakter dan membahas
   sepuluh hal berbeda. Kalau seluruh artikel jadi satu vektor, vektor itu jadi
   rata-rata dari semua topik dan tidak mewakili satu pun dengan tajam.
2. Anggaran konteks. Yang dikirim ke LLM cuma potongan yang relevan, bukan
   seluruh korpus.

Strateginya memecah di batas paragraf, bukan di jumlah karakter tetap. Memotong
di tengah kalimat menghasilkan chunk yang maknanya rusak. Paragraf adalah unit
semantik alami yang sudah disediakan penulisnya secara gratis.

Antar chunk diberi overlap supaya kalimat di perbatasan tidak kehilangan
konteks dari paragraf sebelumnya.

Jalankan:  uv run python -m rag.chunk
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    MIN_CHUNK_CHARS,
    RAW_DIR,
)

# Judul bagian pada ekstrak Wikipedia berbentuk "== Sejarah ==".
HEADING_RE = re.compile(r"^=+\s*(.+?)\s*=+$")


@dataclass
class Chunk:
    """Satu potongan teks yang siap di-embed dan dikembalikan sebagai hasil pencarian."""

    id: str
    doc_id: str
    title: str
    url: str
    section: str
    position: int
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})

    def context_header(self) -> str:
        """Prefiks yang menempelkan chunk ke asalnya.

        Chunk yang berdiri sendiri sering kehilangan rujukan -- "model ini
        dilatih dengan ..." merujuk ke apa? Menempelkan judul artikel dan nama
        bagian di depan teks memulihkan konteks itu, baik untuk model embedding
        maupun untuk manusia yang membaca hasil pencarian.
        """
        return f"{self.title} — {self.section}" if self.section else self.title

    def embed_text(self) -> str:
        """Teks yang benar-benar dikirim ke model embedding."""
        return f"{self.context_header()}\n\n{self.text}"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Memisah teks artikel jadi pasangan (nama_bagian, isi).

    Teks sebelum judul bagian pertama adalah pembuka, diberi nama bagian kosong.
    """
    sections: list[tuple[str, str]] = []
    current_name = ""
    buffer: list[str] = []

    for line in text.splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading:
            if buffer:
                sections.append((current_name, "\n".join(buffer).strip()))
            current_name = heading.group(1)
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_name, "\n".join(buffer).strip()))

    return [(name, body) for name, body in sections if body]


def _merge_runts(packed: list[str]) -> list[str]:
    """Menyatukan chunk kekecilan ke tetangganya.

    Bagian pendek yang isinya cuma satu baris menghasilkan chunk sependek
    beberapa kata. Chunk seperti itu punya embedding yang nyaris acak dan sering
    muncul sebagai hasil palsu, jadi lebih baik digabung.
    """
    merged: list[str] = []
    for text in packed:
        if merged and len(text) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{text}"
        else:
            merged.append(text)

    # Chunk pertama tidak punya tetangga di kiri, jadi digabung ke kanan.
    if len(merged) > 1 and len(merged[0]) < MIN_CHUNK_CHARS:
        merged[1] = f"{merged[0]}\n\n{merged[1]}"
        merged.pop(0)

    return merged


def pack_paragraphs(paragraphs: list[str]) -> list[str]:
    """Menggabungkan paragraf sampai mendekati ukuran target.

    Paragraf yang sendirian sudah melebihi target dibiarkan utuh, bukan dipotong
    paksa -- lebih baik satu chunk kepanjangan daripada satu kalimat terbelah.
    """
    packed: list[str] = []
    buffer: list[str] = []
    size = 0

    for paragraph in paragraphs:
        if buffer and size + len(paragraph) > CHUNK_TARGET_CHARS:
            packed.append("\n\n".join(buffer))
            # Bawa ekor chunk sebelumnya sebagai overlap.
            tail = packed[-1][-CHUNK_OVERLAP_CHARS:]
            buffer = [tail, paragraph] if CHUNK_OVERLAP_CHARS else [paragraph]
            size = sum(len(item) for item in buffer)
        else:
            buffer.append(paragraph)
            size += len(paragraph)

    if buffer:
        packed.append("\n\n".join(buffer))

    return _merge_runts(packed)


def chunk_document(doc: dict) -> list[Chunk]:
    """Mengubah satu dokumen mentah jadi daftar chunk."""
    chunks: list[Chunk] = []

    for section_name, body in split_sections(doc["text"]):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        packed = pack_paragraphs(paragraphs)

        # Bagian yang seluruh isinya kekecilan (mis. "Lihat pula") dilewati:
        # tidak ada tetangga di dalam bagian ini untuk menampungnya.
        if len(packed) == 1 and len(packed[0]) < MIN_CHUNK_CHARS:
            continue

        for text in packed:
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc['id']}::{position}",
                    doc_id=doc["id"],
                    title=doc["title"],
                    url=doc["url"],
                    section=section_name,
                    position=position,
                    text=text,
                )
            )

    return chunks


def load_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise SystemExit(
            f"Tidak ada dokumen di {raw_dir}. Jalankan dulu: uv run python -m rag.fetch_corpus"
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def chunk_corpus(raw_dir: Path = RAW_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(raw_dir):
        chunks.extend(chunk_document(doc))
    return chunks


def main() -> None:
    chunks = chunk_corpus()
    sizes = sorted(chunk.n_chars for chunk in chunks)
    docs = {chunk.doc_id for chunk in chunks}

    print(f"{len(chunks)} chunk dari {len(docs)} dokumen")
    print(f"  terpendek : {sizes[0]:,} karakter")
    print(f"  median    : {sizes[len(sizes) // 2]:,} karakter")
    print(f"  terpanjang: {sizes[-1]:,} karakter")
    print(f"  rata-rata : {sum(sizes) // len(sizes):,} karakter")


if __name__ == "__main__":
    main()
