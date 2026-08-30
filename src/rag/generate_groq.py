"""Penyusunan jawaban lewat Groq (gratis).

Groq memakai API yang kompatibel dengan OpenAI: hanya chat completion, tanpa
tipe content block `document`, dan **tanpa sitasi native**. Itu berarti jaminan
yang dipakai jalur Claude hilang di sini, dan penggantinya harus dibangun
sendiri.

Cara kerjanya:

1. Chunk dinomori `[1]`, `[2]`, ... di dalam prompt.
2. Model diminta menempelkan penanda itu di kalimat yang memakainya.
3. Penanda yang keluar **diverifikasi program**, bukan dipercaya begitu saja.

Langkah ketiga itu intinya. Sitasi berbasis prompt gampang dikarang: model bisa
menulis `[7]` padahal cuma ada 5 dokumen, dan tanpa pemeriksaan itu lolos
kelihatan meyakinkan. Di sini penanda di luar jangkauan dicatat sebagai
`hallucinated_markers` dan dilaporkan ke pengguna.

Yang tetap tidak bisa ditiru dari sitasi native Claude: pemetaan ke rentang
karakter persis di dokumen sumber. Groq cuma bisa memberi tahu dokumen mana,
bukan kalimat mana. Jadi ini pengganti yang jujur, bukan yang setara.
"""

from __future__ import annotations

import re

from rag.config import GROQ_MODEL, groq_api_key
from rag.retrieve import Hit

SYSTEM_PROMPT = """\
Kamu menjawab pertanyaan hanya berdasarkan dokumen bernomor yang diberikan.

Aturan:
- Jawab dalam Bahasa Indonesia yang jelas dan ringkas.
- Pakai hanya informasi dari dokumen. Jangan menambah pengetahuan dari luar.
- Setiap klaim harus diikuti penanda sumbernya, misalnya [1] atau [2][3].
- Pakai hanya nomor dokumen yang benar-benar ada. Jangan mengarang nomor.
- Kalau dokumen tidak memuat jawabannya, katakan terus terang bahwa informasi
  itu tidak ada di dokumen yang tersedia. Jangan menebak.
- Langsung ke jawaban, tanpa basa-basi pembuka.
"""

MAX_TOKENS = 2_000

# Suhu rendah: tugasnya menyarikan dokumen, bukan mengarang. Tidak nol, karena
# nol kadang membuat model tersangkut mengulang frasa yang sama.
TEMPERATURE = 0.2

MARKER_RE = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Menyusun dokumen bernomor beserta pertanyaannya.

    Nomor di sini 1-based dan harus sejalan dengan urutan `hits`, karena itulah
    yang dipakai untuk memetakan penanda balik ke chunk asalnya.
    """
    blocks = [
        f"[{number}] {hit.chunk.context_header()}\n{hit.chunk.text}"
        for number, hit in enumerate(hits, start=1)
    ]
    documents = "\n\n".join(blocks)
    return f"Dokumen:\n\n{documents}\n\nPertanyaan: {question}"


def parse_markers(text: str, n_documents: int) -> tuple[list[int], list[int]]:
    """Memisahkan penanda yang sah dari yang mengada-ada.

    Mengembalikan (indeks_sah_0based, nomor_karangan_1based). Penanda di luar
    jangkauan berarti model menyebut dokumen yang tidak pernah dikirim.
    """
    valid: list[int] = []
    hallucinated: list[int] = []

    for match in MARKER_RE.findall(text):
        number = int(match)
        if 1 <= number <= n_documents:
            if number - 1 not in valid:
                valid.append(number - 1)
        elif number not in hallucinated:
            hallucinated.append(number)

    return sorted(valid), hallucinated


def generate(
    question: str,
    hits: list[Hit],
    api_key: str | None = None,
    model: str = GROQ_MODEL,
    on_text=None,
) -> tuple[str, list[int], list[int]]:
    """Menghasilkan (teks, indeks_dokumen_terkutip, penanda_karangan)."""
    from groq import Groq

    client = Groq(api_key=api_key or groq_api_key())

    stream = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
    )

    parts: list[str] = []
    for event in stream:
        piece = event.choices[0].delta.content
        if piece:
            parts.append(piece)
            if on_text is not None:
                on_text(piece)

    text = "".join(parts).strip()
    cited, hallucinated = parse_markers(text, len(hits))
    return text, cited, hallucinated
