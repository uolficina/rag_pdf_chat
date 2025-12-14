# Terminal PDF RAG

Python CLI that reads a local PDF, splits it into chunks, indexes with FAISS, re-ranks with a CrossEncoder, and answers via Mistral chat. It runs in a loop: pick a PDF, ask questions, generate a quick semantic index, read specific pages, or switch files without restarting. FAISS indexes are cached by file hash inside `index/`, so you can reopen previously processed PDFs without recomputing embeddings.

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

Set your API key:
```bash
export MISTRAL_API_KEY="your-key-here"  # Windows: set MISTRAL_API_KEY=your-key-here
```

## How to run
```bash
python rag.py
```

## Menu flow
1) **Load PDF**: provide a file path (same folder or absolute).  
2) **Chat with the PDF**: ask questions about the loaded document; type `back` to return to the menu. Shows token counts for each answer.  
3) **Generate Semantic Index**: creates short titles per page using Mistral (API usage).  
4) **Show specific page**: enter a page number to view the full extracted text; falls back to chunked view if needed.  
5) **Choose a loaded file from list**: pick a cached FAISS index stored in `index/` (skips re-embedding).  
6) **Exit**: quit the CLI.

Example:
```bash
export MISTRAL_API_KEY="my-key"
python rag.py
# Choose 1) Load PDF -> /path/to/document.pdf
# Choose 2) Chat with the PDF -> "What are the main conclusions?"
```

## Saved indexes
- On the first load of a PDF, its embeddings/metadata are cached in `index/<sha>.faiss` and `index/<sha>.meta.json`; a `manifest.json` tracks the list.
- Option 5 in the menu lists cached files by name and docid prefix so you can reopen them instantly.
- Moving/renaming the PDF is fine; the cache key is the file content hash. Delete files in `index/` to force a rebuild.

## How it works
- Chunking: `split_chunks` makes 2,000-character windows with 400 overlap and records the source page.
- Retrieval: normalized embeddings (`intfloat/multilingual-e5-base`) indexed in FAISS (Inner Product).
- Re-rank: `search_rerank` reranks the top 30 with `cross-encoder/ms-marco-MiniLM-L-6-v2` and returns the best 3.
- Generation: `build_mistral_prompt` + `mistral_chat` send a concise prompt to `mistral-small-latest` and print token usage.
- Direct reading: `show_page` uses the raw extracted page text when available.

## Quick tweaks
- Chunk size/overlap: adjust `chunk_size` and `overlap` in `split_chunks`.
- Candidate counts: tune `k_base` and `k_final` in `search_rerank`.
- Temperature/model: change `temperature` or `mistral_model` in `mistral_chat`.
- Titles/index: `generate_index` calls `generate_chunk_title` (Mistral-powered) to label pages.

## Troubleshooting
- **Missing key**: Runtime error telling you to set `MISTRAL_API_KEY` means the env var is absent or empty.
- **PDF has no extractable text**: `pypdf` needs embedded text; run OCR if the file is image-only.
- **Large PDFs are slow/heavy**: reduce `chunk_size`, limit pages, or load a smaller file.
