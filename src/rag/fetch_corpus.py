"""Tahap 1 — mengambil korpus.

Mengambil 50 artikel Wikipedia Bahasa Indonesia bertema kecerdasan buatan lewat
API resmi MediaWiki, lalu menyimpannya satu file JSON per artikel di data/raw/.

Kenapa disimpan ke disk dan ikut di-commit: korpus adalah *input* sistem RAG,
bukan hasil turunan. Kalau Wikipedia berubah besok, hasil eksperimen kemarin
masih bisa direproduksi. Tiap dokumen menyimpan revision_id supaya versinya
persis bisa dilacak.

Jalankan:  uv run python -m rag.fetch_corpus
"""

from __future__ import annotations

import json
import re
import time

import httpx

from rag.config import (
    MIN_DOC_CHARS,
    RAW_DIR,
    TARGET_DOCS,
    USER_AGENT,
    WIKI_LANG,
    WIKI_QUERIES,
)

API_URL = f"https://{WIKI_LANG}.wikipedia.org/w/api.php"

# Halaman disambiguasi dan daftar isinya cuma tautan, tidak ada prosa untuk dijawab.
SKIP_TITLE_PATTERNS = (
    re.compile(r"^Daftar\b", re.IGNORECASE),
    re.compile(r"\(disambiguasi\)", re.IGNORECASE),
)


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def search_titles(client: httpx.Client, query: str, limit: int = 50) -> list[str]:
    """Mencari judul artikel yang cocok dengan sebuah query."""
    response = client.get(
        API_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srnamespace": 0,
            "format": "json",
        },
    )
    response.raise_for_status()
    return [hit["title"] for hit in response.json()["query"]["search"]]


def fetch_article(client: httpx.Client, title: str) -> dict | None:
    """Mengambil teks polos satu artikel beserta revision id-nya.

    Ekstrak teks penuh (tanpa exintro) dibatasi satu halaman per request oleh
    MediaWiki, jadi memang tidak bisa di-batch.
    """
    response = client.get(
        API_URL,
        params={
            "action": "query",
            "prop": "extracts|revisions|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "inprop": "url",
            "titles": title,
            "format": "json",
        },
    )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page or not page.get("extract"):
        return None

    text = _clean(page["extract"])
    if len(text) < MIN_DOC_CHARS:
        return None

    revision = page.get("revisions", [{}])[0]
    return {
        "id": f"wiki-{page['pageid']}",
        "page_id": page["pageid"],
        "title": page["title"],
        "url": page.get("fullurl", f"https://{WIKI_LANG}.wikipedia.org/?curid={page['pageid']}"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "text": text,
        "n_chars": len(text),
        "license": "CC BY-SA 4.0",
        "source": f"{WIKI_LANG}.wikipedia.org",
    }


def _clean(text: str) -> str:
    """Membuang bagian ekor yang tidak informatif dan merapikan baris kosong."""
    # Bagian setelah "Referensi"/"Pranala luar" isinya sitasi, bukan prosa.
    text = re.split(r"\n=+\s*(Referensi|Pranala luar|Lihat pula|Daftar pustaka|Bacaan lanjutan)\s*=+", text)[0]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_skippable(title: str) -> bool:
    return any(pattern.search(title) for pattern in SKIP_TITLE_PATTERNS)


def collect_candidates(client: httpx.Client) -> list[str]:
    """Menggabungkan hasil semua query jadi satu daftar judul unik, urutan terjaga."""
    seen: set[str] = set()
    candidates: list[str] = []
    for query in WIKI_QUERIES:
        for title in search_titles(client, query):
            if title not in seen and not _is_skippable(title):
                seen.add(title)
                candidates.append(title)
    return candidates


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with _client() as client:
        candidates = collect_candidates(client)
        print(f"{len(candidates)} kandidat judul dari {len(WIKI_QUERIES)} query.")

        saved = 0
        for title in candidates:
            if saved >= TARGET_DOCS:
                break

            doc = fetch_article(client, title)
            time.sleep(0.2)  # sopan terhadap server Wikipedia

            if doc is None:
                print(f"  lewati  {title}")
                continue

            path = RAW_DIR / f"{doc['id']}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            saved += 1
            print(f"  [{saved:2d}/{TARGET_DOCS}] {doc['title']} ({doc['n_chars']:,} karakter)")

    if saved < TARGET_DOCS:
        raise SystemExit(
            f"Hanya dapat {saved} dokumen dari target {TARGET_DOCS}. "
            "Tambahkan query di WIKI_QUERIES atau turunkan MIN_DOC_CHARS."
        )

    print(f"\nSelesai: {saved} dokumen tersimpan di {RAW_DIR}")


if __name__ == "__main__":
    main()
