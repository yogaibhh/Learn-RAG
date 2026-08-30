# rag-wikipedia-id

Project RAG (Retrieval-Augmented Generation) pertama: tanya-jawab di atas 50 artikel
Wikipedia Bahasa Indonesia bertema Kecerdasan Buatan & Machine Learning.

Status: dalam pengembangan. Lihat roadmap di bawah.

## Alur

```
Wikipedia API  ->  chunking  ->  embedding (Jina v5)  ->  index
                                                            |
                        pertanyaan  ->  embedding query  ->  retrieval (dense + BM25)
                                                            |
                                                      jawaban (Claude, opsional)
```

## Roadmap

- [x] Kerangka project
- [ ] Ambil 50 artikel dari Wikipedia
- [ ] Chunking
- [ ] Embedding via Jina v5
- [ ] Index & retrieval hybrid
- [ ] CLI
- [ ] Generation dengan sitasi
- [ ] Streamlit UI
- [ ] Evaluasi retrieval
