# rag-wikipedia-id

Sistem RAG (*Retrieval-Augmented Generation*) di atas 50 artikel Wikipedia
Bahasa Indonesia bertema kecerdasan buatan.

Dibangun untuk dipahami, bukan cuma dipakai. Tiap tahap bisa dijalankan sendiri
dan tiap keputusan desainnya ditulis alasannya di komentar modulnya.

---

## Coba dulu tanpa API key apa pun

Jalur ini tidak butuh key, tidak butuh unduhan model, dan selesai dalam
hitungan detik.

```bash
uv sync
uv run python -m rag.fetch_corpus          # ambil 50 artikel (hanya sekali)
uv run python cli.py index --embedder none # index tanpa vektor
uv run python cli.py ask "siapa pendiri Anthropic"
```

Hasilnya memakai BM25 saja. Itu sudah cukup untuk melihat seluruh pipeline
bekerja dari ujung ke ujung.

## Aktifkan pencarian semantik

Ambil API key Jina gratis (tanpa kartu kredit) di
<https://jina.ai/?sui=apikey>, lalu:

```bash
cp .env.example .env      # isi JINA_API_KEY
uv run python cli.py index
uv run python cli.py compare "cara komputer belajar sendiri dari contoh"
```

`compare` menjalankan dense, BM25, dan hybrid berdampingan untuk pertanyaan
yang sama. Ini perintah paling berguna di repo ini: bedanya kelihatan langsung,
bukan cuma dibaca di teori.

Alternatif tanpa API sama sekali — model dijalankan di mesin sendiri
(unduhan sekitar 2,5 GB):

```bash
uv sync --extra local
uv run python cli.py index --embedder local
```

## Aktifkan penyusunan jawaban

Isi `ANTHROPIC_API_KEY` di `.env`. Tanpa itu, `ask` tetap jalan dan menampilkan
chunk hasil pencarian apa adanya (*mode retrieval-only*).

## Antarmuka web

```bash
uv run python -m streamlit run app.py
```

> Pakai `python -m streamlit`, bukan `streamlit` langsung. Di sebagian mesin
> Windows, shim `streamlit.exe` diblokir kebijakan Application Control
> (`os error 4551`).

---

## Alur pipeline

```
data/raw/*.json          50 artikel, hasil fetch_corpus
        |
        v   chunk.py     pecah di batas paragraf + overlap
474 chunk
        |
        v   embed.py     Jina v5, task=retrieval.passage, late_chunking
data/index/              chunks.jsonl + embeddings.npy + meta.json
        |
        v   retrieve.py  dense (cosine) + BM25, digabung dengan RRF
5 chunk teratas
        |
        v   generate.py  Claude claude-opus-5, sitasi native
jawaban + sitasi terverifikasi
```

| Modul | Tahap |
|---|---|
| `src/rag/fetch_corpus.py` | ambil korpus dari API MediaWiki |
| `src/rag/chunk.py` | pecah dokumen jadi chunk |
| `src/rag/embed.py` | embedding lewat Jina v5 API |
| `src/rag/embed_local.py` | embedding di mesin sendiri (opsional) |
| `src/rag/index.py` | bangun dan muat index |
| `src/rag/retrieve.py` | pencarian dense, BM25, hybrid |
| `src/rag/generate.py` | susun jawaban dengan sitasi |

---

## Keputusan desain

**Embedding asimetris.** Chunk di-embed dengan `task=retrieval.passage`,
pertanyaan dengan `retrieval.query`. Jina memakai adapter berbeda untuk
keduanya, karena "Apa itu overfitting?" memang tidak mirip secara permukaan
dengan paragraf yang menjelaskan overfitting. Memakai task yang sama untuk
keduanya adalah kesalahan yang menurunkan kualitas pencarian tanpa gejala.

**Late chunking.** Semua chunk satu dokumen dikirim dalam satu request, model
membacanya sebagai satu kesatuan, lalu embedding tiap chunk diambil. Chunk yang
berbunyi "model ini dilatih dengan ..." jadi tidak kehilangan acuan. Syaratnya:
satu request = satu dokumen, karena server menggabungkan seluruh input.

**RRF, bukan penjumlahan skor.** Skor cosine berkisar 0..1 sementara skor BM25
tidak punya batas atas. Dijumlahkan langsung, BM25 selalu menang. Reciprocal
Rank Fusion hanya memakai peringkat, jadi kedua metode punya suara setara.

**Sitasi dari API, bukan dari prompt.** Menempelkan "sebutkan sumbernya" di
prompt menghasilkan sitasi yang ditulis model dari ingatan, dan model bisa
mengarang nomor sumber yang meyakinkan. Di sini tiap chunk dikirim sebagai
content block `document` dengan `citations` aktif, jadi API mengembalikan
potongan teks persis yang mendasari tiap kalimat.

**Numpy, bukan vector database.** 474 chunk. Perkalian matriks 474×1024 selesai
dalam mikrodetik. Index aproksimasi seperti HNSW belum ada gunanya di skala ini
dan hanya menyembunyikan mekanismenya.

**Prompt caching sengaja tidak dipakai.** Yang masuk prompt cuma chunk hasil
retrieval, yang berubah tiap pertanyaan. Bagian yang stabil hanya system
prompt, dan panjangnya di bawah ambang minimum agar sebuah prefix bisa
di-cache. Ini contoh bagus kapan caching justru *tidak* membantu.

---

## Evaluasi

```bash
uv run python -m eval.run --misses
```

15 pertanyaan dengan dokumen jawaban yang sudah ditandai. Hasil dengan BM25
saja:

| mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| acak (baseline) | 0% | 0% | 13% | 0,033 |
| bm25 | 93% | 100% | 100% | 0,956 |

Baseline acak ikut dihitung karena angka tanpa pembanding tidak berarti apa-apa.

> **Catatan jujur:** sebagian pertanyaan uji memakai kata yang mirip judul
> artikelnya, jadi angka ini menguntungkan BM25. Untuk perbandingan yang adil
> terhadap dense, set pertanyaan ini perlu ditambah kasus parafrase yang tidak
> berbagi kata dengan dokumennya.

---

## Uji

```bash
uv run pytest
```

46 uji: pemecah dokumen, tokenisasi, tiga mode pencarian, rumus RRF secara
numerik, dan uji asap antarmuka Streamlit lewat `AppTest`.

---

## Lisensi dan atribusi

- **Korpus** di `data/raw/` berasal dari Wikipedia Bahasa Indonesia,
  berlisensi **CC BY-SA 4.0**. Tiap berkas menyimpan `url` dan `revision_id`
  supaya versi persisnya bisa dilacak.
- **Bobot model Jina v5** (jalur `--embedder local`) berlisensi
  **CC BY-NC 4.0 — non-komersial**. Untuk belajar seperti project ini aman;
  untuk produk komersial perlu izin Jina. Jalur API punya ketentuan sendiri.
- Model lokal dimuat dengan `trust_remote_code=True`, artinya kode dari repo
  Hugging Face-nya dijalankan di mesin kamu.

---

## Latihan lanjutan

1. Tambah pertanyaan parafrase ke `eval/questions.jsonl`, lalu ukur ulang —
   di situ dense seharusnya mulai mengungguli BM25.
2. Coba dimensi Matryoshka yang lebih kecil (`JINA_DIMENSIONS = 256`) dan lihat
   berapa banyak kualitas yang hilang dibanding hemat memorinya.
3. Ubah `CHUNK_TARGET_CHARS` dan amati efeknya ke hit@k. Chunk kecil menaikkan
   presisi tapi memutus konteks.
4. Tambahkan tahap *reranking* di atas hasil hybrid.
5. Ganti index numpy dengan vector database sungguhan, lalu bandingkan
   kecepatannya setelah korpus diperbesar sepuluh kali lipat.
