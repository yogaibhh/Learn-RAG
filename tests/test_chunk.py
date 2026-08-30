"""Uji unit untuk pemecah dokumen."""

from __future__ import annotations

import pytest

from rag.chunk import (
    Chunk,
    chunk_document,
    pack_paragraphs,
    split_sections,
)
from rag.config import CHUNK_TARGET_CHARS, MIN_CHUNK_CHARS


def make_doc(text: str) -> dict:
    return {
        "id": "wiki-1",
        "title": "Kecerdasan Buatan",
        "url": "https://id.wikipedia.org/wiki/Kecerdasan_buatan",
        "text": text,
    }


class TestSplitSections:
    def test_teks_tanpa_judul_jadi_satu_bagian_tanpa_nama(self):
        assert split_sections("Sebuah paragraf.") == [("", "Sebuah paragraf.")]

    def test_memisah_pada_judul_bagian(self):
        text = "Pembuka.\n\n== Sejarah ==\n\nIsi sejarah.\n\n== Penerapan ==\n\nIsi penerapan."
        assert split_sections(text) == [
            ("", "Pembuka."),
            ("Sejarah", "Isi sejarah."),
            ("Penerapan", "Isi penerapan."),
        ]

    def test_judul_bertingkat_ikut_dikenali(self):
        assert split_sections("=== Sub ===\n\nIsi.") == [("Sub", "Isi.")]

    def test_bagian_kosong_dibuang(self):
        text = "== Kosong ==\n\n== Berisi ==\n\nAda isinya."
        assert split_sections(text) == [("Berisi", "Ada isinya.")]


class TestPackParagraphs:
    def test_paragraf_pendek_digabung_jadi_satu_chunk(self):
        paragraphs = ["A" * 300, "B" * 300, "C" * 300]
        assert len(pack_paragraphs(paragraphs)) == 1

    def test_memecah_saat_melewati_target(self):
        paragraphs = ["A" * 700, "B" * 700, "C" * 700]
        packed = pack_paragraphs(paragraphs)
        assert len(packed) > 1

    def test_paragraf_raksasa_tidak_dipotong_paksa(self):
        giant = "A" * (CHUNK_TARGET_CHARS * 3)
        packed = pack_paragraphs([giant])
        assert packed == [giant]

    def test_ada_overlap_antar_chunk(self):
        paragraphs = ["A" * 700, "B" * 700, "C" * 700]
        packed = pack_paragraphs(paragraphs)
        # Ekor chunk pertama harus muncul lagi di awal chunk kedua.
        assert packed[1].startswith(packed[0][-50:])

    def test_chunk_kekecilan_digabung_ke_tetangga(self):
        paragraphs = ["A" * 1100, "B" * 20]
        packed = pack_paragraphs(paragraphs)
        assert all(len(chunk) >= MIN_CHUNK_CHARS for chunk in packed)

    def test_daftar_kosong_menghasilkan_nol_chunk(self):
        assert pack_paragraphs([]) == []


class TestChunkDocument:
    def test_id_chunk_unik_dan_berurutan(self):
        doc = make_doc("== Satu ==\n\n" + "A" * 2000 + "\n\n== Dua ==\n\n" + "B" * 2000)
        chunks = chunk_document(doc)
        assert [c.id for c in chunks] == [f"wiki-1::{i}" for i in range(len(chunks))]

    def test_metadata_sumber_ikut_terbawa(self):
        chunks = chunk_document(make_doc("== Sejarah ==\n\n" + "A" * 500))
        assert chunks[0].title == "Kecerdasan Buatan"
        assert chunks[0].section == "Sejarah"
        assert chunks[0].doc_id == "wiki-1"
        assert chunks[0].url.endswith("Kecerdasan_buatan")

    def test_bagian_yang_seluruhnya_kekecilan_dilewati(self):
        doc = make_doc("== Isi ==\n\n" + "A" * 500 + "\n\n== Lihat pula ==\n\nDaftar.")
        sections = {c.section for c in chunk_document(doc)}
        assert sections == {"Isi"}

    def test_semua_chunk_memenuhi_ukuran_minimum(self):
        doc = make_doc("== Isi ==\n\n" + "\n\n".join("A" * 400 for _ in range(10)))
        assert all(c.n_chars >= MIN_CHUNK_CHARS for c in chunk_document(doc))


class TestChunkModel:
    @pytest.fixture
    def chunk(self) -> Chunk:
        return Chunk(
            id="wiki-1::0",
            doc_id="wiki-1",
            title="Kecerdasan Buatan",
            url="https://example.org",
            section="Sejarah",
            position=0,
            text="Isi chunk.",
        )

    def test_context_header_menggabungkan_judul_dan_bagian(self, chunk):
        assert chunk.context_header() == "Kecerdasan Buatan — Sejarah"

    def test_context_header_tanpa_bagian_hanya_judul(self, chunk):
        chunk.section = ""
        assert chunk.context_header() == "Kecerdasan Buatan"

    def test_embed_text_menyertakan_header_dan_isi(self, chunk):
        embedded = chunk.embed_text()
        assert embedded.startswith("Kecerdasan Buatan — Sejarah")
        assert "Isi chunk." in embedded

    def test_bolak_balik_json(self, chunk):
        import json

        assert Chunk.from_dict(json.loads(chunk.to_json())) == chunk
