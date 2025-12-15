import hashlib
import json
import os

import faiss

from rag.config import BASE_DIR, state


def doc_id(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def doc_path(docid):
    idx_path = os.path.join(BASE_DIR, f"{docid}.faiss")
    meta_path = os.path.join(BASE_DIR, f"{docid}.meta.json")
    manifest_path = os.path.join(BASE_DIR, "manifest.json")
    return idx_path, meta_path, manifest_path


def save_index(docid, index, meta):
    idx_path, meta_path, _ = doc_path(docid)
    faiss.write_index(index, idx_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_index(docid):
    idx_path, meta_path, _ = doc_path(docid)
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index not found")
    index = faiss.read_index(idx_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return index, meta


def save_manifest(docid, file_path, title=None, total_pages=None):
    _, _, manifest_path = doc_path(docid)
    entry = {
        "docid": docid,
        "name": os.path.basename(file_path),
        "size": os.path.getsize(file_path),
        "mtime": os.path.getmtime(file_path),
        "title": title,
        "total_pages": total_pages,
    }
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest = [m for m in manifest if m["docid"] != docid]
    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def list_docs():
    _, _, manifest_path = doc_path("dummy")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return json.load(f)


def load_doc(docid):
    index, meta = load_index(docid)
    state["index"] = index
    state["chunk_texts"] = meta["chunk_texts"]
    state["chunks"] = meta["chunks"]
    # Backward compatibility: older metas used "page_text" instead of "page_texts".
    state["page_texts"] = meta.get("page_texts") or meta.get("page_text") or []
    state["total_pages"] = meta["total_pages"]
    return meta
