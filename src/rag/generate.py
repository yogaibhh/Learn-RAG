"""Tahap 6 — menyusun jawaban dari chunk yang ditemukan.

Bagian "G" dari RAG: chunk hasil retrieval dikirim ke model bahasa, yang lalu
menjawab hanya berdasarkan isi chunk itu.

Ada dua penyedia, dan bedanya justru terletak pada seberapa bisa sitasinya
dipercaya.

**Groq** (`generate_groq.py`) -- gratis, jadi ini bawaan project. API-nya
kompatibel dengan OpenAI: hanya chat completion, tanpa sitasi native. Sitasinya
memakai penanda `[1]`, `[2]` di dalam teks, lalu **diverifikasi program**
terhadap daftar dokumen yang benar-benar dikirim. Nomor di luar jangkauan
dicatat sebagai karangan alih-alih diloloskan.

**Claude** -- berbayar, tapi sitasinya tidak lewat prompt sama sekali. Tiap
chunk dikirim sebagai content block bertipe `document` dengan `citations`
aktif, sehingga API mengembalikan potongan teks persis yang mendasari tiap
kalimat, lengkap dengan posisi karakternya.

Perbedaannya nyata dan sengaja tidak disamarkan: verifikasi penanda hanya bisa
menjawab "dokumen ini pernah dikirim atau tidak", sementara sitasi native
menjawab "kalimat mana persisnya". Yang pertama menangkap nomor karangan; yang
kedua juga menangkap salah rujuk ke dokumen yang benar.

Tanpa key mana pun, modul ini tidak error. Sistem berjalan di mode
retrieval-only: chunk yang ditemukan ditampilkan apa adanya. Untuk belajar RAG
itu justru berguna, karena mutu jawaban akhir sepenuhnya dibatasi mutu
retrieval -- kalau chunk-nya salah, jawaban sebagus apa pun tetap salah.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import CLAUDE_MODEL, anthropic_api_key, groq_api_key
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
    """Hasil akhir: teks jawaban, sitasi, dan chunk sumbernya.

    Dua tingkat kehalusan sitasi, tergantung penyedia:

    `citations`
        Tingkat karakter, hanya tersedia lewat Claude. Berisi potongan teks
        persis yang mendasari tiap kalimat, dikembalikan API.

    `cited_indices`
        Tingkat dokumen, tersedia di kedua jalur. Menunjuk posisi di `hits`.

    `hallucinated_markers`
        Khusus jalur Groq: nomor sumber yang disebut model tapi tidak pernah
        dikirim kepadanya. Selalu kosong di jalur Claude, karena di sana sitasi
        datang dari API dan bukan dari teks yang ditulis model.
    """

    text: str
    hits: list[Hit]
    citations: list[Citation] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    hallucinated_markers: list[int] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    refused: bool = False
    refusal_reason: str | None = None

    @property
    def generated(self) -> bool:
        """False kalau ini cuma hasil retrieval tanpa tahap generation."""
        return self.model is not None

    @property
    def has_verified_citations(self) -> bool:
        """True hanya kalau sitasinya datang dari API, bukan dari teks model."""
        return bool(self.citations)

    def cited_hits(self) -> list[Hit]:
        """Chunk yang benar-benar dipakai dalam jawaban.

        Berbeda dari `hits`, yang berisi semua chunk yang dikirim ke model.
        Selisih keduanya menarik: chunk yang dikirim tapi tidak pernah dikutip
        adalah tanda retrieval mengambil sesuatu yang ternyata tidak terpakai.
        """
        used = set(self.cited_indices)
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
        "Mode retrieval-only: tidak ada GROQ_API_KEY maupun ANTHROPIC_API_KEY.",
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


def choose_provider(provider: str = "auto") -> str:
    """Memilih penyedia generation yang tersedia.

    Groq didahulukan karena gratis, dan project ini memang ditujukan untuk
    dijalankan tanpa biaya. Claude dipakai kalau keynya ada dan Groq tidak,
    atau kalau diminta eksplisit -- di sana sitasinya lebih kuat.
    """
    if provider != "auto":
        return provider
    if groq_api_key():
        return "groq"
    if anthropic_api_key():
        return "claude"
    return "none"


def answer(
    question: str,
    hits: list[Hit],
    api_key: str | None = None,
    provider: str = "auto",
    on_text=None,
) -> Answer:
    """Menyusun jawaban dari chunk yang ditemukan.

    `on_text` dipanggil untuk tiap potongan teks yang masuk, supaya jawaban bisa
    ditampilkan sambil mengalir alih-alih menunggu selesai.
    """
    chosen = choose_provider(provider)

    if chosen == "none":
        return retrieval_only_answer(question, hits)

    if not hits:
        return Answer(
            text="Tidak ada chunk yang cocok, jadi tidak ada bahan untuk menjawab.",
            hits=[],
        )

    if chosen == "groq":
        return _answer_with_groq(question, hits, api_key, on_text)

    return _answer_with_claude(question, hits, api_key, on_text)


def _answer_with_groq(question, hits, api_key, on_text) -> Answer:
    from rag.config import GROQ_MODEL
    from rag.generate_groq import generate

    text, cited, hallucinated = generate(question, hits, api_key=api_key, on_text=on_text)
    return Answer(
        text=text,
        hits=hits,
        cited_indices=cited,
        hallucinated_markers=hallucinated,
        provider="groq",
        model=GROQ_MODEL,
    )


def _answer_with_claude(question, hits, api_key, on_text) -> Answer:
    key = api_key or anthropic_api_key()

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
            provider="claude",
            model=CLAUDE_MODEL,
            refused=True,
            refusal_reason=getattr(details, "explanation", None) if details else None,
        )

    text, citations = _extract(message)
    return Answer(
        text=text,
        hits=hits,
        citations=citations,
        cited_indices=sorted({citation.document_index for citation in citations}),
        provider="claude",
        model=CLAUDE_MODEL,
    )
