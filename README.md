# Terminal PDF RAG

Python CLI to chat with local PDFs. It extracts text, creates overlapped chunks, builds embeddings (E5), indexes with FAISS, reranks with a CrossEncoder, and answers via Mistral chat. Everything runs in a menu loop: load a PDF, ask questions, generate semantic page titles, read specific pages, or reopen saved indexes.

## Requirements
- Python 3.10+
- Environment variable `MISTRAL_API_KEY` with your Mistral key
- Dependencies: `pypdf`, `sentence_transformers`, `faiss-cpu`, `mistralai`, `numpy`

## Quick setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pypdf sentence_transformers faiss-cpu mistralai numpy
```

Set the key:
```bash
export MISTRAL_API_KEY="your-key"  # Windows: set MISTRAL_API_KEY=your-key
```

## Running
```bash
python main.py
```

## Menu flow (rag/cli.py)
1) **Load PDF**: enter the PDF path (relative or absolute).  
2) **Chat with the PDF**: ask questions; `back` returns to the menu. Shows token counts.  
3) **Generate Semantic Index**: requests short titles (2–5 words) per page using Mistral.  
4) **Show specific page**: shows the raw page text; if missing, prints the chunks on that page.  
5) **Choose a loaded file from list**: lists saved FAISS indexes and reopens without reprocessing.  
6) **Exit**: quit.

Quick example:
```bash
export MISTRAL_API_KEY="my-key"
python main.py
# 1) Load PDF -> /path/to/file.pdf
# 2) Chat with the PDF -> "What are the main conclusions?"
```

## Persistence and reuse (rag/index_store.py)
- Each PDF produces `index/<sha>.faiss` (vectors) + `index/<sha>.meta.json` (chunks, pages, texts).
- `index/manifest.json` stores the list of processed documents.
- The identifier is the SHA1 of the file contents; moving/renaming the PDF does not break the cache.
- Delete files in `index/` to force reprocessing.

## Architecture overview
- **Config**: `rag/config.py` centralizes models and the shared `state` dict.
- **Models**: `rag/embeddings.py` lazily loads `intfloat/multilingual-e5-base` and the CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Pipeline**: `rag/retrieval.py` reads the PDF (`pypdf`), builds chunks (2,000 chars, 400 overlap), creates normalized embeddings, and indexes with FAISS (Inner Product). It searches top-30, reranks top-3 with the CrossEncoder, builds a prompt, and calls `mistral-small-latest`.
- **Direct reading**: `rag/pdf_utils.py` shows full page text when available.

## Quick tweaks
- Chunk size/overlap: edit `split_chunks` in `rag/pdf_utils.py`.
- Candidate counts: `k_base` and `k_final` in `search_rerank`.
- Temperature/model: adjust `mistral_chat`.
- Index folder: `BASE_DIR` in `rag/config.py` (default `index`).

## Troubleshooting
- **Missing `MISTRAL_API_KEY`**: the app raises an error asking for the variable; set it and retry.
- **PDF has no extractable text**: image-only PDFs need prior OCR.
- **High memory/time**: reduce `chunk_size`, limit pages, or use smaller PDFs.
