"""Antarmuka web Streamlit.

    uv run streamlit run app.py

Yang ditampilkan bukan cuma jawabannya. Skor tiap chunk, peringkatnya di dense
dan di BM25, serta chunk mana yang akhirnya benar-benar dikutip semuanya
ditunjukkan -- karena bagian itulah yang mengajari cara kerja RAG. Jawaban akhir
justru bagian yang paling tidak informatif kalau kamu sedang belajar.
"""

from __future__ import annotations

import streamlit as st

from rag import index as index_module
from rag.embed import EmbeddingError
from rag.generate import answer
from rag.retrieve import Retriever

st.set_page_config(page_title="RAG Wikipedia Indonesia", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Memuat index ...")
def load_retriever() -> tuple[Retriever, str | None]:
    """Memuat index dan embedder satu kali, lalu dipakai ulang antar interaksi.

    Mengembalikan pesan peringatan kalau embedder gagal disiapkan, supaya
    penyebabnya bisa ditampilkan ke pengguna alih-alih hilang diam-diam.
    """
    index = index_module.load()
    name = index.meta.get("embedder")

    if not name:
        return Retriever(index, None), None

    try:
        if str(name).startswith("jinaai/"):
            from rag.embed_local import LocalEmbedder

            return Retriever(index, LocalEmbedder()), None
        from rag.embed import JinaEmbedder

        return Retriever(index, JinaEmbedder()), None
    except EmbeddingError as exc:
        return Retriever(index, None), str(exc)


def render_hit(hit, cited: bool) -> None:
    mark = "✅ dikutip" if cited else "— tidak dikutip"
    label = f"{hit.rank}. {hit.chunk.context_header()}  ·  skor {hit.score:.4f}  ·  {mark}"
    with st.expander(label, expanded=False):
        st.caption(f"Asal peringkat: {hit.why()}")
        st.write(hit.chunk.text)
        st.caption(hit.chunk.url)


def main() -> None:
    st.title("📚 RAG Wikipedia Bahasa Indonesia")
    st.caption("Tanya-jawab di atas 50 artikel bertema kecerdasan buatan.")

    try:
        retriever, warning = load_retriever()
    except SystemExit as exc:
        st.error(str(exc))
        st.stop()

    if warning:
        st.warning(f"{warning}\n\nSementara ini pencarian memakai BM25 saja.")

    with st.sidebar:
        st.header("Pengaturan")

        available = ["hybrid", "dense", "bm25"] if retriever.can_dense else ["bm25"]
        mode = st.radio("Mode pencarian", available, help="Coba mode berbeda untuk pertanyaan yang sama.")
        k = st.slider("Jumlah chunk", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("Index")
        st.write(retriever.index.describe())
        for key, value in retriever.index.meta.items():
            st.caption(f"{key}: {value}")

        if not retriever.can_dense:
            st.info(
                "Index ini belum punya vektor, jadi hanya BM25 yang tersedia.\n\n"
                "Bangun index bervektor:\n\n"
                "`uv run python cli.py index`"
            )

    question = st.text_input(
        "Pertanyaan",
        placeholder="Apa bedanya pembelajaran mesin dan pembelajaran mendalam?",
    )

    if not question:
        st.info("Tulis pertanyaan di atas untuk mulai.")
        return

    hits = retriever.search(question, k=k, mode=mode)

    if not hits:
        st.warning("Tidak ada chunk yang cocok dengan pertanyaan ini.")
        return

    st.subheader("Jawaban")
    placeholder = st.empty()
    buffer: list[str] = []

    def on_text(text: str) -> None:
        buffer.append(text)
        placeholder.markdown("".join(buffer))

    with st.spinner("Menyusun jawaban ..."):
        result = answer(question, hits, on_text=on_text)

    if not result.generated:
        placeholder.empty()
        st.info(
            "Mode retrieval-only: ANTHROPIC_API_KEY belum diisi, jadi tahap "
            "penyusunan jawaban dilewati. Chunk hasil pencarian ada di bawah."
        )
    elif result.refused:
        placeholder.empty()
        st.error(f"Permintaan ditolak. {result.refusal_reason or ''}".strip())
    else:
        placeholder.markdown(result.text)

    if result.citations:
        st.subheader(f"Sitasi ({len(result.citations)})")
        st.caption("Dikembalikan langsung oleh API, bukan ditulis model dari ingatan.")
        for citation in result.citations:
            st.markdown(f"**{citation.document_title}**")
            st.markdown(f"> {' '.join(citation.cited_text.split())}")

    cited_ids = {hit.chunk.id for hit in result.cited_hits()}
    st.subheader(f"Chunk yang ditemukan ({len(hits)})")
    if result.generated and result.citations:
        st.caption(f"{len(cited_ids)} dari {len(hits)} chunk benar-benar terpakai dalam jawaban.")

    for hit in hits:
        render_hit(hit, cited=hit.chunk.id in cited_ids)


if __name__ == "__main__":
    main()
