"""Konfigurasi terpusat: path, konstanta, dan pembacaan variabel lingkungan."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Path -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"

# --- Korpus -----------------------------------------------------------------

WIKI_LANG = "id"
TARGET_DOCS = 50

# Beberapa query, bukan satu, supaya korpusnya tidak menumpuk di satu subtopik.
WIKI_QUERIES = [
    "kecerdasan buatan",
    "pembelajaran mesin",
    "jaringan saraf tiruan",
    "pemrosesan bahasa alami",
    "pembelajaran mendalam",
]

# Artikel di bawah ini terlalu pendek untuk jadi bahan tanya-jawab yang berarti.
MIN_DOC_CHARS = 1500

# Wikipedia mewajibkan User-Agent yang deskriptif dan bisa dihubungi.
USER_AGENT = "rag-wikipedia-id/0.1 (https://github.com/yogaibhh/Learn-RAG)"

# --- Chunking ---------------------------------------------------------------

CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# --- Embedding --------------------------------------------------------------

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"
JINA_DIMENSIONS = 1024
LOCAL_MODEL = "jinaai/jina-embeddings-v5-text-nano"

# --- Generation -------------------------------------------------------------

CLAUDE_MODEL = "claude-opus-5"


def jina_api_key() -> str | None:
    return os.getenv("JINA_API_KEY") or None


def anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or None
