"""Uji untuk tahap penyusunan jawaban.

Fokusnya pada lapisan verifikasi sitasi, bukan pada mutu jawaban model. Bagian
itulah yang jadi pengaman: kalau `parse_markers` salah, penanda karangan lolos
tanpa ketahuan dan seluruh klaim "sumber terverifikasi" jadi bohong.

Tidak ada uji di sini yang memanggil API. Semuanya murni fungsi.
"""

from __future__ import annotations

import pytest

from rag.chunk import Chunk
from rag.generate import Answer, Citation, build_document_blocks, retrieval_only_answer
from rag.generate_groq import build_prompt, parse_markers
from rag.retrieve import Hit


def make_hit(rank: int, title: str, text: str = "Isi dokumen.") -> Hit:
    chunk = Chunk(
        id=f"wiki-{rank}::0",
        doc_id=f"wiki-{rank}",
        title=title,
        url=f"https://id.wikipedia.org/wiki/{title}",
        section="Sejarah",
        position=0,
        text=text,
    )
    return Hit(chunk=chunk, score=1.0 / rank, rank=rank)


@pytest.fixture
def hits() -> list[Hit]:
    return [make_hit(1, "Anthropic"), make_hit(2, "OpenAI"), make_hit(3, "ChatGPT")]


class TestParseMarkers:
    """Inti pengaman jalur Groq."""

    def test_penanda_sah_jadi_indeks_nol_based(self):
        valid, hallucinated = parse_markers("Klaim A [1] dan klaim B [3].", n_documents=3)
        assert valid == [0, 2]
        assert hallucinated == []

    def test_penanda_di_luar_jangkauan_ditandai_karangan(self):
        valid, hallucinated = parse_markers("Menurut [7] hal ini benar.", n_documents=3)
        assert valid == []
        assert hallucinated == [7]

    def test_memisahkan_yang_sah_dari_yang_karangan(self):
        valid, hallucinated = parse_markers("Ada [2] tapi juga [9].", n_documents=3)
        assert valid == [1]
        assert hallucinated == [9]

    def test_penanda_nol_dianggap_karangan(self):
        # Penomoran di prompt 1-based, jadi [0] tidak pernah sah.
        _, hallucinated = parse_markers("Lihat [0].", n_documents=3)
        assert hallucinated == [0]

    def test_penanda_berulang_tidak_diduplikasi(self):
        valid, _ = parse_markers("[1] lalu [1] lagi [1].", n_documents=3)
        assert valid == [0]

    def test_penanda_berdempetan_terbaca_semua(self):
        valid, _ = parse_markers("Keduanya benar [1][2].", n_documents=3)
        assert valid == [0, 1]

    def test_teks_tanpa_penanba_menghasilkan_kosong(self):
        assert parse_markers("Tidak ada rujukan sama sekali.", n_documents=3) == ([], [])

    def test_indeks_selalu_terurut(self):
        valid, _ = parse_markers("[3] lalu [1] lalu [2].", n_documents=3)
        assert valid == [0, 1, 2]


class TestBuildPrompt:
    def test_dokumen_dinomori_satu_based(self, hits):
        prompt = build_prompt("Pertanyaan?", hits)
        assert "[1] Anthropic" in prompt
        assert "[2] OpenAI" in prompt
        assert "[3] ChatGPT" in prompt

    def test_penomoran_sejalan_dengan_urutan_hits(self, hits):
        # Nomor prompt harus memetakan balik ke hits lewat parse_markers.
        prompt = build_prompt("Pertanyaan?", hits)
        valid, _ = parse_markers("Menurut [2].", n_documents=len(hits))
        assert hits[valid[0]].chunk.title == "OpenAI"
        assert "[2] OpenAI" in prompt

    def test_pertanyaan_ikut_masuk(self, hits):
        assert "Siapa pendirinya?" in build_prompt("Siapa pendirinya?", hits)


class TestBuildDocumentBlocks:
    def test_sitasi_diaktifkan_di_semua_blok(self, hits):
        blocks = build_document_blocks(hits)
        assert all(block["citations"] == {"enabled": True} for block in blocks)

    def test_urutan_blok_sama_dengan_urutan_hits(self, hits):
        blocks = build_document_blocks(hits)
        assert [block["title"].split(" — ")[0] for block in blocks] == [
            "Anthropic",
            "OpenAI",
            "ChatGPT",
        ]

    def test_sumber_bertipe_teks_polos(self, hits):
        source = build_document_blocks(hits)[0]["source"]
        assert source["type"] == "text"
        assert source["media_type"] == "text/plain"


class TestAnswer:
    def test_cited_hits_menyaring_lewat_cited_indices(self, hits):
        result = Answer(text="x", hits=hits, cited_indices=[0, 2])
        assert [hit.chunk.title for hit in result.cited_hits()] == ["Anthropic", "ChatGPT"]

    def test_generated_false_tanpa_model(self, hits):
        assert Answer(text="x", hits=hits).generated is False

    def test_generated_true_dengan_model(self, hits):
        assert Answer(text="x", hits=hits, model="apa-saja").generated is True

    def test_sitasi_terverifikasi_hanya_kalau_ada_citations(self, hits):
        tanpa = Answer(text="x", hits=hits, cited_indices=[0], provider="groq")
        dengan = Answer(
            text="x",
            hits=hits,
            citations=[Citation("kutipan", 0, "Anthropic", 0, 7)],
            provider="claude",
        )
        assert tanpa.has_verified_citations is False
        assert dengan.has_verified_citations is True


class TestRetrievalOnly:
    def test_menyertakan_semua_chunk(self, hits):
        result = retrieval_only_answer("Pertanyaan?", hits)
        assert all(hit.chunk.title in result.text for hit in hits)

    def test_ditandai_belum_digenerate(self, hits):
        assert retrieval_only_answer("Pertanyaan?", hits).generated is False

    def test_tanpa_hasil_memberi_pesan_jelas(self):
        assert "Tidak ada chunk" in retrieval_only_answer("Pertanyaan?", []).text
