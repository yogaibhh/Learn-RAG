"""Uji asap untuk antarmuka Streamlit.

Memakai AppTest bawaan Streamlit, yang menjalankan skripnya sungguhan tanpa
membuka peramban. Tujuannya menangkap kesalahan pemakaian API Streamlit yang
kalau tidak diuji baru ketahuan saat halaman dibuka manusia.
"""

from __future__ import annotations

import pytest

from rag import index as index_module
from rag.config import PROJECT_ROOT

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# AppTest menyelesaikan path relatif terhadap direktori kerja, yang berbeda-beda
# tergantung dari mana pytest dipanggil. Path absolut menghindari itu.
APP_PATH = str(PROJECT_ROOT / "app.py")


@pytest.fixture(scope="module")
def app():
    try:
        index_module.load()
    except SystemExit:
        pytest.skip("Index belum dibangun. Jalankan: uv run python cli.py index")
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def test_halaman_terbuka_tanpa_galat(app):
    assert list(app.exception) == []


def test_judul_tampil(app):
    assert "RAG Wikipedia Bahasa Indonesia" in app.title[0].value


def test_pengaturan_mode_tersedia_di_sidebar(app):
    modes = app.sidebar.radio[0].options
    assert "bm25" in modes


def test_pertanyaan_menghasilkan_chunk(app):
    app.text_input[0].set_value("siapa pendiri Anthropic").run()
    assert list(app.exception) == []
    assert any("Chunk yang ditemukan" in sub.value for sub in app.subheader)
    assert len(app.expander) > 0


def test_hasil_teratas_relevan(app):
    app.text_input[0].set_value("siapa pendiri Anthropic").run()
    assert "Amodei" in app.expander[0].label or "Anthropic" in app.expander[0].label
