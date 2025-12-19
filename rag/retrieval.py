import os
import time

import faiss
import numpy as np
from mistralai import Mistral

from rag.config import (
    ANSWER_LANG,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    state,
)
from rag.embeddings import get_models
from rag.index_store import (
    doc_id,
    doc_path,
    load_index,
    save_index,
    save_manifest,
)
from rag.pdf_utils import load_pdf, split_chunks


def _resolve_language(preferred_lang=None):
    lang = (preferred_lang or state.get("answer_lang") or ANSWER_LANG or "").strip().lower()
    if lang in ("pt", "pt-br", "portugues", "português"):
        return "pt"
    if lang in ("en", "en-us", "english"):
        return "en"
    return ""


def build_mistral_prompt(question, contexts, preferred_lang=None):
    lang = _resolve_language(preferred_lang)
    lang_line = ""
    if lang == "pt":
        lang_line = "Responda em português do Brasil.\n"
    elif lang == "en":
        lang_line = "Answer in English.\n"

    blocks = []
    for ctx in contexts:
        blocks.append(f"[Page {ctx['page']}] {ctx['text']}")
    context_text = "\n\n".join(blocks)
    return (
        f"{lang_line}"
        "Summarize or answer the question using only the provided context. "
        "If there is no answer, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}"
    )


def mistral_chat(question, contexts, preferred_lang=None):
    if state.get("offline"):
        raise RuntimeError("Offline mode enabled; LLM call skipped")
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set the MISTRAL_API_KEY environment variable")
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt_text = build_mistral_prompt(question, contexts, preferred_lang=preferred_lang)
    resp = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.2,
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage


def offline_chat(contexts, snippet=400):
    if not contexts:
        return "[Offline] No context found.", None
    lines = ["[Offline] No call to LLM. Relevant Chunks:"]
    for i, ctx in enumerate(contexts, start=1):
        text = (ctx.get("text") or "").strip()
        if len(text) > snippet:
            text = text[:snippet].rstrip() + "..."
        page = ctx.get("page", "?")
        score = ctx.get("score")
        score_txt = f" (score {score:.3f})" if score is not None else ""
        lines.append(f"{i}) Página {page}{score_txt}: {text}")
    return "\n".join(lines), None


def search_rerank(question, k_base=30, k_final=3):
    embed_model, cross = get_models()
    if state["index"] is None:
        raise RuntimeError("Load a PDF first")
    results = []
    question_fmt = f"query: {question}"
    q_emb = embed_model.encode([question_fmt], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, idxs = state["index"].search(q_emb, k_base)
    pairs = [(question, state["chunk_texts"][i]) for i in idxs[0]]
    rerank_scores = cross.predict(pairs)
    reranked = sorted(zip(rerank_scores, idxs[0]), key=lambda x: x[0], reverse=True)[:k_final]
    for score, i in reranked:
        results.append({
            "score": float(score),
            "page": state["chunks"][i]["page"],
            "text": state["chunks"][i]["text"],
        })
    return results


def generate_chunk_title(chunk_text, attempts=5, base_delay=2):
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = (
        "Generate ONLY ONE SHORT TITLE for the text below.\n"
        "Title rules:\n"
        "- It must have between 2 and 5 words.\n"
        "- Do not describe long sentences.\n"
        "- Do not write explanations.\n"
        "- Do not use colons.\n\n"
        f"Text:\n{chunk_text}"
    )

    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            status = getattr(e, "http_status", None) or getattr(e, "status_code", None)
            if (status == 429) or ("rate limit" in msg) or ("too many requests" in msg):
                if attempt == attempts:
                    raise
                wait_time = base_delay * (2 ** (attempt - 1))
                print(f"Rate limit; waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time)
                continue
            raise


def generate_index(chunks):
    displayed_pages = set()
    for c in chunks:
        page = c["page"]
        if page in displayed_pages:
            continue
        displayed_pages.add(page)
        chunk_text = c["text"]
        title = generate_chunk_title(chunk_text)
        print(f"Page {page}: {title}")


def prepare_document(file_path):
    """
    Load an existing FAISS index if present; otherwise build from the PDF.
    Updates the shared state and returns it.
    """
    docid = doc_id(file_path)
    idx_path, meta_path, _ = doc_path(docid)

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index, meta = load_index(docid)
        state["index"] = index
        state["chunk_texts"] = meta["chunk_texts"]
        state["chunks"] = meta["chunks"]
        state["page_texts"] = meta.get("page_texts") or meta.get("page_text") or []
        state["raw_page_texts"] = meta.get("raw_page_texts") or []
        state["total_pages"] = meta["total_pages"]
        get_models()
        return state

    pdf_text, offsets, page_texts, raw_page_texts = load_pdf(file_path)
    state["total_pages"] = len(offsets)
    state["page_texts"] = page_texts
    state["raw_page_texts"] = raw_page_texts
    chunks = split_chunks(pdf_text, offsets, chunk_size=2000, overlap=400)
    state["chunks"] = chunks
    state["chunk_texts"] = [c["text"] for c in chunks]

    embed_model, _ = get_models()
    embed_texts = [f"passage: {txt}" for txt in state["chunk_texts"]]
    embs = embed_model.encode(embed_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    state["index"] = index

    meta = {
        "chunk_texts": state["chunk_texts"],
        "chunks": chunks,
        "page_texts": page_texts,
        "raw_page_texts": raw_page_texts,
        "total_pages": state["total_pages"],
    }
    save_index(docid, index, meta)
    save_manifest(docid, file_path, total_pages=state["total_pages"])
    return state
