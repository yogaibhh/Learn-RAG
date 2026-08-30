"""Uji untuk pencarian.

Sebagian uji memakai index tiruan supaya cepat dan tidak bergantung jaringan.
Sisanya berjalan di atas korpus asli untuk memastikan pipeline-nya benar-benar
menyambung dari ujung ke ujung.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag import index as index_module
from rag.chunk import Chunk
from rag.index import Index
from rag.retrieve import RRF_K, Retriever, tokenize


def make_chunk(chunk_id: str, title: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        title=title,
        url="https://example.org",
        section="",
        position=0,
        text=text,
    )


@pytest.fixture
def toy_index() -> Index:
    chunks = [
        make_chunk("d1::0", "Pembelajaran Mesin", "Algoritma yang belajar dari data latih."),
        make_chunk("d2::0", "AlphaGo", "AlphaGo mengalahkan Lee Sedol pada permainan Go."),
        make_chunk("d3::0", "Fotosintesis", "Tumbuhan mengubah cahaya matahari jadi energi."),
    ]
    return Index(chunks=chunks, embeddings=None, meta={})


class FakeEmbedder:
    """Embedder tiruan: query diarahkan ke satu chunk tertentu."""

    name = "fake"
    dimensions = 3
    supports_late_chunking = False

    def __init__(self, target: int = 0):
        self.target = target

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[self.target] = 1.0
        return vector

    def embed_passages(self, texts, late_chunking: bool = False) -> np.ndarray:
        return np.eye(len(texts), self.dimensions, dtype=np.float32)


class TestTokenize:
    def test_huruf_dikecilkan(self):
        assert tokenize("AlphaGo MENANG") == ["alphago", "menang"]

    def test_stopword_dibuang(self):
        assert "yang" not in tokenize("model yang belajar")

    def test_tanda_baca_dibuang(self):
        assert tokenize("Apa itu overfitting?") == ["overfitting"]

    def test_token_satu_huruf_dibuang(self):
        assert tokenize("a b komputer") == ["komputer"]


class TestBm25:
    def test_menemukan_nama_diri(self, toy_index):
        hits = Retriever(toy_index).bm25("AlphaGo", k=1)
        assert hits[0].chunk.title == "AlphaGo"

    def test_peringkat_dimulai_dari_satu(self, toy_index):
        hits = Retriever(toy_index).bm25("data", k=3)
        assert [hit.rank for hit in hits] == [1, 2, 3]

    def test_k_membatasi_jumlah_hasil(self, toy_index):
        assert len(Retriever(toy_index).bm25("data", k=2)) == 2

    def test_k_lebih_besar_dari_korpus_tidak_error(self, toy_index):
        assert len(Retriever(toy_index).bm25("data", k=99)) == 3

    def test_skor_menurun(self, toy_index):
        hits = Retriever(toy_index).bm25("AlphaGo Lee Sedol", k=3)
        assert hits == sorted(hits, key=lambda hit: -hit.score)


class TestDense:
    def test_tanpa_vektor_menolak_dengan_jelas(self, toy_index):
        with pytest.raises(RuntimeError, match="dense"):
            Retriever(toy_index, FakeEmbedder()).dense("apa saja", k=1)

    def test_menemukan_chunk_yang_vektornya_searah(self, toy_index):
        toy_index.embeddings = np.eye(3, dtype=np.float32)
        hits = Retriever(toy_index, FakeEmbedder(target=1)).dense("apa saja", k=1)
        assert hits[0].chunk.title == "AlphaGo"

    def test_dense_rank_terisi(self, toy_index):
        toy_index.embeddings = np.eye(3, dtype=np.float32)
        hits = Retriever(toy_index, FakeEmbedder()).dense("apa saja", k=2)
        assert [hit.dense_rank for hit in hits] == [1, 2]


class TestHybrid:
    def test_turun_ke_bm25_kalau_tidak_ada_vektor(self, toy_index):
        hits = Retriever(toy_index).hybrid("AlphaGo", k=1)
        assert hits[0].chunk.title == "AlphaGo"

    def test_skor_rrf_sesuai_rumus(self, toy_index):
        toy_index.embeddings = np.eye(3, dtype=np.float32)
        hits = Retriever(toy_index, FakeEmbedder(target=1)).hybrid("AlphaGo", k=3)
        top = hits[0]
        # Peringkat 1 di kedua metode: 1/(k+1) + 1/(k+1).
        assert top.chunk.title == "AlphaGo"
        assert top.score == pytest.approx(2 / (RRF_K + 1))

    def test_hasil_kedua_metode_digabung_bukan_ditumpuk(self, toy_index):
        toy_index.embeddings = np.eye(3, dtype=np.float32)
        hits = Retriever(toy_index, FakeEmbedder(target=2)).hybrid("AlphaGo", k=3)
        assert len({hit.chunk.id for hit in hits}) == len(hits)

    def test_why_menjelaskan_asal_peringkat(self, toy_index):
        toy_index.embeddings = np.eye(3, dtype=np.float32)
        top = Retriever(toy_index, FakeEmbedder(target=1)).hybrid("AlphaGo", k=1)[0]
        assert "dense #" in top.why() and "bm25 #" in top.why()


class TestSearchDispatch:
    def test_mode_tidak_dikenal_ditolak(self, toy_index):
        with pytest.raises(ValueError, match="Mode tidak dikenal"):
            Retriever(toy_index).search("apa saja", mode="ajaib")

    @pytest.mark.parametrize("mode", ["bm25", "hybrid"])
    def test_mode_yang_didukung_tanpa_vektor(self, toy_index, mode):
        assert Retriever(toy_index).search("AlphaGo", k=1, mode=mode)


@pytest.fixture(scope="module")
def real_index() -> Index:
    """Index sungguhan dari disk. Uji yang memakainya dilewati kalau belum ada."""
    try:
        return index_module.load()
    except SystemExit:
        pytest.skip("Index belum dibangun. Jalankan: uv run python cli.py index")


class TestKorpusAsli:
    """Uji di atas index sungguhan, memastikan pipeline menyambung ujung ke ujung."""

    def test_korpus_berisi_lima_puluh_dokumen(self, real_index):
        assert len({chunk.doc_id for chunk in real_index.chunks}) == 50

    def test_pertanyaan_tentang_pendiri_menemukan_orangnya(self, real_index):
        hits = Retriever(real_index).bm25("siapa pendiri Anthropic", k=5)
        assert any("Amodei" in hit.chunk.title or "Anthropic" in hit.chunk.title for hit in hits)

    def test_istilah_teknis_menemukan_artikel_yang_tepat(self, real_index):
        hits = Retriever(real_index).bm25("pembelajaran mesin", k=5)
        assert any("embelajaran" in hit.chunk.title for hit in hits)

    def test_setiap_hasil_membawa_tautan_sumber(self, real_index):
        for hit in Retriever(real_index).bm25("kecerdasan buatan", k=5):
            assert hit.chunk.url.startswith("https://")
