"""Antarmuka baris perintah.

    uv run python cli.py index                  bangun index (butuh JINA_API_KEY)
    uv run python cli.py index --embedder none  bangun index tanpa vektor, BM25 saja
    uv run python cli.py search "pertanyaan"    cari chunk yang relevan
    uv run python cli.py compare "pertanyaan"   bandingkan dense, bm25, dan hybrid
    uv run python cli.py info                   lihat isi index sekarang
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag import index as index_module
from rag.embed import EmbeddingError
from rag.retrieve import Hit, Retriever


def build_embedder(kind: str):
    """Memilih embedder sesuai permintaan pengguna.

    'none' mengembalikan None, yang bikin index terbangun tanpa vektor dan
    pencarian jatuh ke BM25. Itu jalur tanpa API key sama sekali.
    """
    if kind == "none":
        return None
    if kind == "jina":
        from rag.embed import JinaEmbedder

        return JinaEmbedder()
    if kind == "local":
        from rag.embed_local import LocalEmbedder

        return LocalEmbedder()
    raise ValueError(f"Embedder tidak dikenal: {kind!r}")


def open_retriever(embedder_kind: str = "auto") -> Retriever:
    """Memuat index dan menyiapkan pencari.

    Mode 'auto' membaca meta index: kalau index dibangun dengan vektor, embedder
    yang sama dipakai lagi untuk meng-embed pertanyaan. Memakai embedder berbeda
    dari yang dipakai saat indexing akan menghasilkan pencarian yang kacau,
    karena kedua model menempatkan makna di ruang vektor yang berbeda.
    """
    index = index_module.load()

    if embedder_kind == "auto":
        name = index.meta.get("embedder")
        if not name:
            return Retriever(index, None)
        embedder_kind = "local" if str(name).startswith("jinaai/") else "jina"

    try:
        embedder = build_embedder(embedder_kind)
    except EmbeddingError as exc:
        print(f"Peringatan: {exc}\n", file=sys.stderr)
        print("Melanjutkan dengan BM25 saja.\n", file=sys.stderr)
        embedder = None

    return Retriever(index, embedder)


def print_hits(hits: list[Hit], width: int = 88) -> None:
    if not hits:
        print("  (tidak ada hasil)")
        return

    for hit in hits:
        chunk = hit.chunk
        snippet = " ".join(chunk.text.split())[:260]
        print(f"\n  {hit.rank}. {chunk.context_header()}")
        print(f"     skor {hit.score:.4f}   ({hit.why()})")
        for line in textwrap.wrap(snippet + " ...", width=width - 5):
            print(f"     {line}")
        print(f"     {chunk.url}")


def cmd_index(args: argparse.Namespace) -> int:
    try:
        embedder = build_embedder(args.embedder)
    except EmbeddingError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        index_module.build(embedder)
    finally:
        if embedder is not None:
            embedder.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    retriever = open_retriever(args.embedder)
    mode = args.mode
    if mode in ("dense", "hybrid") and not retriever.can_dense:
        print(
            f"Mode '{mode}' butuh index bervektor; index ini tidak punya. "
            "Memakai BM25.\n",
            file=sys.stderr,
        )
        mode = "bm25"

    print(f"Pertanyaan : {args.query}")
    print(f"Mode       : {mode}")
    print_hits(retriever.search(args.query, k=args.k, mode=mode))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Menjalankan ketiga mode berdampingan.

    Ini perintah paling berguna untuk belajar: di sinilah kelihatan kapan BM25
    menang dan kapan dense menang untuk pertanyaan yang sama.
    """
    retriever = open_retriever(args.embedder)
    modes = ["bm25"] if not retriever.can_dense else ["dense", "bm25", "hybrid"]

    if len(modes) == 1:
        print(
            "Index ini tidak punya vektor, jadi tidak ada yang bisa dibandingkan.\n"
            "Bangun index dengan embedder:  uv run python cli.py index\n",
            file=sys.stderr,
        )

    print(f"Pertanyaan: {args.query}\n")
    for mode in modes:
        print("=" * 88)
        print(f"MODE: {mode}")
        print("=" * 88)
        print_hits(retriever.search(args.query, k=args.k, mode=mode))
        print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Pipeline RAG utuh: cari chunk, lalu susun jawaban darinya."""
    from rag.generate import answer

    from rag.generate import choose_provider

    retriever = open_retriever(args.embedder)
    mode = args.mode if retriever.can_dense else "bm25"
    hits = retriever.search(args.query, k=args.k, mode=mode)

    provider = choose_provider(args.provider)
    print(f"Pertanyaan : {args.query}")
    print(f"Retrieval  : {mode}, {len(hits)} chunk")
    print(f"Generation : {provider}\n")
    print("-" * 88)

    result = answer(
        args.query,
        hits,
        provider=args.provider,
        on_text=lambda text: print(text, end="", flush=True),
    )

    # Dalam mode retrieval-only teksnya belum tercetak, karena tidak ada aliran.
    if not result.generated:
        print(result.text)
    print("\n" + "-" * 88)

    if result.refused:
        print(f"\nDitolak. {result.refusal_reason or ''}".rstrip())
        return 1

    # Penanda yang menunjuk dokumen di luar yang dikirim: model mengarang nomor.
    if result.hallucinated_markers:
        markers = ", ".join(f"[{n}]" for n in result.hallucinated_markers)
        print(
            f"\n⚠  Model menyebut sumber {markers}, padahal cuma ada "
            f"{len(hits)} dokumen yang dikirim. Penanda itu karangan."
        )

    if result.citations:
        print(f"\nSitasi ({len(result.citations)}), dikembalikan API dan bukan karangan model:")
        for citation in result.citations:
            quote = " ".join(citation.cited_text.split())[:110]
            print(f"  - {citation.document_title}")
            print(f'      "{quote}..."')

    used = result.cited_hits()
    if used:
        kind = "terverifikasi API" if result.has_verified_citations else "penanda tervalidasi"
        print(f"\nSumber terpakai ({kind}): {len(used)} dari {len(hits)} chunk yang dikirim.")
        for index, hit in enumerate(result.hits):
            if index in set(result.cited_indices):
                print(f"  [{index + 1}] {hit.chunk.context_header()}")
                print(f"      {hit.chunk.url}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    index = index_module.load()
    print(index.describe())
    print()
    for key, value in index.meta.items():
        print(f"  {key:16} {value}")

    titles = sorted({chunk.title for chunk in index.chunks})
    print(f"\n  {len(titles)} artikel:")
    for title in titles:
        print(f"    - {title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="RAG di atas 50 artikel Wikipedia Bahasa Indonesia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="bangun index dari data/raw/")
    p_index.add_argument(
        "--embedder",
        choices=["jina", "local", "none"],
        default="jina",
        help="jina = API (butuh JINA_API_KEY), local = model di mesin sendiri, "
        "none = tanpa vektor, hanya BM25",
    )
    p_index.set_defaults(func=cmd_index)

    for name, handler, help_text in (
        ("search", cmd_search, "cari chunk yang relevan"),
        ("ask", cmd_ask, "pipeline RAG utuh: cari lalu jawab"),
        ("compare", cmd_compare, "bandingkan dense, bm25, dan hybrid berdampingan"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("query", help="pertanyaan atau kata kunci")
        sub.add_argument("-k", type=int, default=5, help="jumlah hasil (default 5)")
        sub.add_argument(
            "--embedder",
            choices=["auto", "jina", "local", "none"],
            default="auto",
            help="default 'auto' mengikuti embedder yang dipakai saat index dibangun",
        )
        if name in ("search", "ask"):
            sub.add_argument(
                "--mode",
                choices=["hybrid", "dense", "bm25"],
                default="hybrid",
                help="metode pencarian (default hybrid)",
            )
        if name == "ask":
            sub.add_argument(
                "--provider",
                choices=["auto", "groq", "claude", "none"],
                default="auto",
                help="penyedia jawaban; 'auto' memilih Groq kalau keynya ada, "
                "lalu Claude, lalu mode retrieval-only",
            )
        sub.set_defaults(func=handler)

    p_info = subparsers.add_parser("info", help="tampilkan ringkasan index")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
