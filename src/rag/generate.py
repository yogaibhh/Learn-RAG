"""Tahap 6 — menyusun jawaban dari chunk yang ditemukan.

Bagian "G" dari RAG. Chunk hasil retrieval dikirim ke Claude, lalu Claude
menjawab hanya berdasarkan isi chunk itu.

Yang membedakan implementasi ini dari kebanyakan tutorial RAG: sitasinya tidak
diminta lewat prompt. Menempelkan "sebutkan sumbernya ya" di prompt menghasilkan
sitasi yang ditulis model dari ingatan -- dan model bisa saja mengarang nomor
sumber yang kelihatan meyakinkan. Di sini tiap chunk dikirim sebagai content
block bertipe `document` dengan `citations: {"enabled": True}`, sehingga API
sendiri yang mengembalikan potongan teks persis yang mendasari tiap kalimat,
lengkap dengan posisi karakternya. Sitasi jadi data terverifikasi, bukan
kalimat yang kebetulan berbentuk sitasi.

Tanpa ANTHROPIC_API_KEY, modul ini tidak error. Sistem berjalan di mode
retrieval-only: chunk yang ditemukan ditampilkan apa adanya. Untuk belajar RAG
itu justru berguna, karena mutu jawaban akhir sepenuhnya dibatasi mutu
retrieval -- kalau chunk-nya salah, jawaban sebagus apa pun tetap salah.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import CLAUDE_MODEL, anthropic_api_key
from rag.retrieve import Hit

SYSTEM_PROMPT = """\
Kamu menjawab pertanyaan hanya berdasarkan dokumen yang diberikan.

Aturan:
- Jawab dalam Bahasa Indonesia yang jelas dan ringkas.
- Pakai hanya informasi dari dokumen. Jangan menambah pengetahuan dari luar.
- Kalau dokumen tidak memuat jawabannya, katakan terus terang bahwa informasi
  itu tidak ada di dokumen yang tersedia. Jangan menebak.
- Kalau dokumen saling bertentangan, sebutkan pertentangannya.
- Langsung ke jawaban, tanpa basa-basi pembuka.
"""

# Batas atas token keluaran. Angkanya longgar karena penagihan mengikuti token
# yang benar-benar dihasilkan, bukan batas ini; panjang jawaban dikendalikan
# lewat instruksi di system prompt.
MAX_TOKENS = 16_000


@dataclass
class Citation:
    """Satu rujukan yang dikembalikan API, bukan yang dikarang model."""

    cited_text: str
    document_index: int
    document_title: str
    start_char_index: int
    end_char_index: int

    @classmethod
    def from_api(cls, data: dict) -> "Citation":
        return cls(
            cited_text=data["cited_text"],
            document_index=data["document_index"],
            document_title=data.get("document_title") or "",
            start_char_index=data.get("start_char_index", 0),
            end_char_index=data.get("end_char_index", 0),
        )


@dataclass
class Answer:
    """Hasil akhir: teks jawaban, sitasi terverifikasi, dan chunk sumbernya."""

    text: str
    hits: list[Hit]
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    refused: bool = False
    refusal_reason: str | None = None

    @property
    def generated(self) -> bool:
        """False kalau ini cuma hasil retrieval tanpa tahap generation."""
        return self.model is not None

    def cited_hits(self) -> list[Hit]:
        """Chunk yang benar-benar dipakai dalam jawaban.

        Berbeda dari `hits`, yang berisi semua chunk yang dikirim ke model.
        Selisih keduanya menarik: chunk yang dikirim tapi tidak pernah dikutip
        adalah tanda retrieval mengambil sesuatu yang ternyata tidak terpakai.
        """
        used = {citation.document_index for citation in self.citations}
        return [hit for index, hit in enumerate(self.hits) if index in used]


def build_document_blocks(hits: list[Hit]) -> list[dict]:
    """Mengubah hasil retrieval jadi content block bertipe document.

    Urutan daftar ini penting: `document_index` pada sitasi merujuk ke posisi di
    sini, jadi urutannya tidak boleh diubah setelah request dikirim.
    """
    return [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": hit.chunk.text,
            },
            "title": hit.chunk.context_header(),
            "context": f"Artikel Wikipedia Bahasa Indonesia. Sumber: {hit.chunk.url}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def retrieval_only_answer(question: str, hits: list[Hit]) -> Answer:
    """Jawaban pengganti saat tahap generation tidak tersedia."""
    if not hits:
        return Answer(text="Tidak ada chunk yang cocok dengan pertanyaan ini.", hits=[])

    lines = [
        "Mode retrieval-only (ANTHROPIC_API_KEY belum diisi).",
        f"Berikut {len(hits)} chunk paling relevan untuk: {question}",
        "",
    ]
    for hit in hits:
        snippet = " ".join(hit.chunk.text.split())
        lines.append(f"[{hit.rank}] {hit.chunk.context_header()}  (skor {hit.score:.4f})")
        lines.append(f"    {snippet[:400]}...")
        lines.append(f"    {hit.chunk.url}")
        lines.append("")

    return Answer(text="\n".join(lines).rstrip(), hits=hits)


def _extract(message) -> tuple[str, list[Citation]]:
    """Memisahkan teks dan sitasi dari balasan API.

    Balasan yang bersitasi terpecah jadi banyak text block: yang mendukung
    sebuah klaim membawa array `citations`, penghubung antar klaim tidak.
    """
    parts: list[str] = []
    citations: list[Citation] = []

    for block in message.content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            data = citation if isinstance(citation, dict) else citation.model_dump()
            if data.get("type") == "char_location":
                citations.append(Citation.from_api(data))

    return "".join(parts), citations


def answer(
    question: str,
    hits: list[Hit],
    api_key: str | None = None,
    on_text=None,
) -> Answer:
    """Menyusun jawaban dari chunk yang ditemukan.

    `on_text` dipanggil untuk tiap potongan teks yang masuk, supaya jawaban bisa
    ditampilkan sambil mengalir alih-alih menunggu selesai.
    """
    key = api_key or anthropic_api_key()
    if not key:
        return retrieval_only_answer(question, hits)

    if not hits:
        return Answer(
            text="Tidak ada chunk yang cocok, jadi tidak ada bahan untuk menjawab.",
            hits=[],
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=key)

    content = build_document_blocks(hits)
    content.append({"type": "text", "text": question})

    # Streaming dipakai supaya jawaban muncul bertahap dan request panjang tidak
    # menabrak batas waktu HTTP.
    #
    # `fallbacks` mengaktifkan pengalihan sisi server: kalau pengklasifikasi
    # keamanan menolak sebuah request, server mengalihkannya sendiri alih-alih
    # mengembalikan galat mentah.
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": content}],
    ) as stream:
        if on_text is not None:
            for chunk in stream.text_stream:
                on_text(chunk)
        message = stream.get_final_message()

    # stop_reason harus diperiksa sebelum membaca isi: pada penolakan, content
    # tidak memuat jawaban yang bisa dipakai.
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        return Answer(
            text="Permintaan ini ditolak oleh pengklasifikasi keamanan.",
            hits=hits,
            model=CLAUDE_MODEL,
            refused=True,
            refusal_reason=getattr(details, "explanation", None) if details else None,
        )

    text, citations = _extract(message)
    return Answer(text=text, hits=hits, citations=citations, model=CLAUDE_MODEL)
