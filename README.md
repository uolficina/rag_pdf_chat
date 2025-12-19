# Terminal PDF RAG

Python CLI to chat with local PDFs. It extracts text, creates overlapped chunks, builds embeddings (E5), indexes with FAISS, reranks with a CrossEncoder, e opcionalmente chama Mistral chat. Tudo roda em um menu: carregar PDF, perguntar, gerar títulos semânticos, ler páginas, reabrir índices, escolher idioma da resposta ou usar modo offline (sem LLM).

## Requirements
- Python 3.10+
- Environment variable `MISTRAL_API_KEY` with your Mistral key
- Dependencies: see `requirements.txt` (Docling for PDF/OCR, RapidOCR + ONNXRuntime, Torch/Sentence-Transformers, FAISS, Mistral client)
- First run needs internet to download Docling/RapidOCR/transformer models. If offline, pre-download and place them in the caches (`~/.cache/docling`, `~/.cache/rapidocr` or `.venv/lib/python*/site-packages/rapidocr/inference_engine/models`, Hugging Face cache).
- For offline replies (sem LLM), você ainda precisa dos modelos locais (embeddings/cross-encoder) já baixados; se não tiver, use BM25/TF-IDF como fallback.

## Quick setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set the key:
```bash
export MISTRAL_API_KEY="your-key"  # Windows: set MISTRAL_API_KEY=your-key
# (opcional) Idioma padrão da resposta: pt ou en
export ANSWER_LANG=pt
# (opcional) Modo offline por padrão (sem LLM)
export OFFLINE_MODE=1
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
6) **Choose answer language**: PT, EN ou automático para próxima respostas.  
7) **Toggle offline mode**: liga/desliga modo sem LLM (mostra apenas trechos relevantes).  
8) **Exit**: quit.

Quick example:
```bash
export MISTRAL_API_KEY="my-key"
python main.py
# 1) Load PDF -> /path/to/file.pdf
# 2) Chat with the PDF -> "What are the main conclusions?"
```

## Persistence and reuse (rag/index_store.py)
- Each PDF produces `index/<sha>.faiss` (vectors) + `index/<sha>.meta.json` (chunks, pages, texts, raw texts).
- `index/manifest.json` stores the list of processed documents.
- The identifier is the SHA1 of the file contents; moving/renaming the PDF does not break the cache.
- Delete files in `index/` to force reprocessing.

## Architecture overview
- **Config**: `rag/config.py` centralizes models and the shared `state` dict (inclui idioma e flag offline).
- **Models**: `rag/embeddings.py` lazily loads `intfloat/multilingual-e5-base` and the CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Pipeline**: `rag/retrieval.py` reads the PDF via Docling (with OCR for image-only PDFs), normalizes preservando parágrafos, armazena texto bruto por página, cria chunks (2,000 chars, 400 overlap), embeddings e índice FAISS. Busca top-30, reranka top-3 com CrossEncoder, monta prompt (com instrução de idioma) e chama `mistral-small-latest` — ou retorna apenas contexto no modo offline.
- **Direct reading**: `rag/pdf_utils.py` shows full page text when available (normalizado ou bruto se preferir).

## Quick tweaks
- Chunk size/overlap: edit `split_chunks` in `rag/pdf_utils.py`.
- Candidate counts: `k_base` and `k_final` in `search_rerank`.
- Temperature/model: adjust `mistral_chat`.
- Index folder: `BASE_DIR` in `rag/config.py` (default `index`).
- Idioma padrão: `ANSWER_LANG` env ou opção 6 no menu.
- Modo offline padrão: `OFFLINE_MODE` env; opção 7 no menu alterna durante a sessão.

## Troubleshooting
- **Missing `MISTRAL_API_KEY`**: the app raises an error asking for the variable; set it and retry.
- **Offline mode**: sem chave ou com offline ativado, o chat exibe apenas os trechos relevantes; não há contagem de tokens.
- **PDF has no extractable text**: Docling runs OCR (RapidOCR + ONNXRuntime). Ensure OCR models exist in the cache (`~/.cache/rapidocr` or `.venv/lib/python*/site-packages/rapidocr/inference_engine/models`) or allow network once to download.
- **High memory/time**: reduce `chunk_size`, limit pages, or use smaller PDFs.
