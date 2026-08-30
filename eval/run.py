"""Evaluasi retrieval.

Kalau retrieval-nya salah, tidak ada model bahasa sebagus apa pun yang bisa
menyelamatkan jawabannya. Jadi bagian inilah yang paling pantas diukur, dan
mengukurnya tidak butuh API key sama sekali.

Dua metrik yang dipakai:

hit@k
    Berapa persen pertanyaan yang dokumen benarnya muncul di antara k hasil
    teratas. Menjawab: "apakah bahannya berhasil terambil?"

MRR
    Mean Reciprocal Rank -- rata-rata dari 1/peringkat dokumen benar pertama.
    Menjawab pertanyaan yang lebih halus: "seberapa tinggi posisinya?" Dokumen
    benar di peringkat 1 bernilai 1,0; di peringkat 5 bernilai 0,2. hit@5 tidak
    membedakan keduanya, MRR membedakan.

Sebagai pembanding disertakan baseline acak. Angka tanpa pembanding tidak ada
artinya: hit@5 sebesar 60% terdengar bagus sampai kamu tahu menebak acak pun
dapat 10%.

Jalankan:  uv run python -m eval.run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import index as index_module  # noqa: E402
from rag.embed import EmbeddingError  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"


@dataclass
class Result:
    mode: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    misses: list[str]


def load_questions() -> list[dict]:
    lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_relevant_rank(hits, relevant: set[str]) -> int | None:
    """Peringkat dokumen benar pertama, atau None kalau tidak ada.

    Beberapa chunk bisa berasal dari dokumen yang sama, jadi yang dinilai adalah
    peringkat pertama kali dokumen benar muncul -- bukan berapa banyak chunknya.
    """
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.doc_id in relevant:
            return rank
    return None


def evaluate(retriever: Retriever, questions: list[dict], mode: str, k: int = 5) -> Result:
    ranks: list[int | None] = []
    misses: list[str] = []

    for question in questions:
        hits = retriever.search(question["question"], k=k, mode=mode)
        rank = first_relevant_rank(hits, set(question["relevant_docs"]))
        ranks.append(rank)
        if rank is None:
            misses.append(f"{question['id']} — {question['question']}")

    total = len(ranks)
    return Result(
        mode=mode,
        hit_at_1=sum(1 for r in ranks if r is not None and r <= 1) / total,
        hit_at_3=sum(1 for r in ranks if r is not None and r <= 3) / total,
        hit_at_5=sum(1 for r in ranks if r is not None and r <= 5) / total,
        mrr=sum(1.0 / r for r in ranks if r is not None) / total,
        misses=misses,
    )


def random_baseline(retriever: Retriever, questions: list[dict], k: int = 5, seed: int = 0) -> Result:
    """Baseline: ambil k chunk acak. Ini lantai yang harus dilewati."""
    rng = random.Random(seed)
    chunks = retriever.index.chunks
    ranks: list[int | None] = []

    for question in questions:
        relevant = set(question["relevant_docs"])
        sample = rng.sample(chunks, min(k, len(chunks)))
        rank = next(
            (i for i, chunk in enumerate(sample, start=1) if chunk.doc_id in relevant), None
        )
        ranks.append(rank)

    total = len(ranks)
    return Result(
        mode="acak (baseline)",
        hit_at_1=sum(1 for r in ranks if r is not None and r <= 1) / total,
        hit_at_3=sum(1 for r in ranks if r is not None and r <= 3) / total,
        hit_at_5=sum(1 for r in ranks if r is not None and r <= 5) / total,
        mrr=sum(1.0 / r for r in ranks if r is not None) / total,
        misses=[],
    )


def print_table(results: list[Result]) -> None:
    print(f"{'mode':<18} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'MRR':>7}")
    print("-" * 50)
    for result in results:
        print(
            f"{result.mode:<18} "
            f"{result.hit_at_1:>6.0%} {result.hit_at_3:>7.0%} "
            f"{result.hit_at_5:>7.0%} {result.mrr:>7.3f}"
        )


def build_retriever(embedder_kind: str) -> Retriever:
    index = index_module.load()

    if embedder_kind == "auto":
        name = index.meta.get("embedder")
        if not name:
            return Retriever(index, None)
        embedder_kind = "local" if str(name).startswith("jinaai/") else "jina"

    try:
        if embedder_kind == "jina":
            from rag.embed import JinaEmbedder

            return Retriever(index, JinaEmbedder())
        if embedder_kind == "local":
            from rag.embed_local import LocalEmbedder

            return Retriever(index, LocalEmbedder())
    except EmbeddingError as exc:
        print(f"Peringatan: {exc}\nMelanjutkan tanpa dense.\n", file=sys.stderr)

    return Retriever(index, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluasi retrieval.")
    parser.add_argument("-k", type=int, default=5, help="jumlah hasil yang dinilai (default 5)")
    parser.add_argument(
        "--embedder", choices=["auto", "jina", "local", "none"], default="auto"
    )
    parser.add_argument("--misses", action="store_true", help="tampilkan pertanyaan yang meleset")
    args = parser.parse_args(argv)

    questions = load_questions()
    retriever = build_retriever(args.embedder)

    print(f"{len(questions)} pertanyaan, k={args.k}")
    print(f"Index: {retriever.index.describe()}\n")

    modes = ["dense", "bm25", "hybrid"] if retriever.can_dense else ["bm25"]
    results = [random_baseline(retriever, questions, args.k)]
    results += [evaluate(retriever, questions, mode, args.k) for mode in modes]

    print_table(results)

    if not retriever.can_dense:
        print(
            "\nHanya BM25 yang diukur: index ini belum punya vektor.\n"
            "Bangun index bervektor untuk membandingkan ketiga mode:\n"
            "  uv run python cli.py index"
        )

    if args.misses:
        for result in results[1:]:
            if result.misses:
                print(f"\nMeleset di mode {result.mode}:")
                for miss in result.misses:
                    print(f"  - {miss}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
