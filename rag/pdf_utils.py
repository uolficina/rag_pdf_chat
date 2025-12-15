from pypdf import PdfReader
import textwrap


def load_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    offsets = []
    cursor = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""
        offsets.append(cursor)
        texts.append(page_text)
        cursor += len(page_text) + 1
    full_text = "\n".join(texts)
    return full_text, offsets, texts


def find_page(start_idx, page_offsets):
    page = 0
    for i, off in enumerate(page_offsets):
        if start_idx < off:
            break
        page = i
    return page


def split_chunks(text, page_offsets, chunk_size=2000, overlap=400):
    if chunk_size <= overlap:
        raise ValueError("Document smaller than the minimum chunk allowed")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_text = text[start:end]
        page = find_page(start, page_offsets)
        chunks.append({"page": page, "text": chunk_text})
    return chunks


def format_text(text, width=80):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width))
    return "\n".join(lines)


def show_page(state, human_page, trim=None):
    page_idx = human_page - 1
    total_pages = state["total_pages"]
    page_texts = state["page_texts"]

    if total_pages and (page_idx < 0 or page_idx >= total_pages):
        print(f"The document has only {total_pages} pages.")
        return

    if page_texts and 0 <= page_idx < len(page_texts):
        text = page_texts[page_idx]
        if trim and len(text) > trim:
            text = text[:trim].rstrip() + "..."
        print(f"\nPage {human_page} (full text):\n{text}")
        return

    page_chunks = [c for c in state["chunks"] if c["page"] == page_idx]
    if not page_chunks:
        print(f"No chunk found on page {human_page}")
        return
    for i, c in enumerate(page_chunks, 1):
        text = format_text(c["text"], width=80)
        if trim and len(text) > trim:
            text = text[:trim].rstrip() + "..."
        print(f"\nChunk {i} (page {human_page}):\n{text}")


def choose_page(state):
    chosen_page = input("Enter the page number you want to read (e.g., 1, 2, 3...): ")
    try:
        num = int(chosen_page)
        if num <= 0:
            raise ValueError
    except ValueError:
        print("Enter a valid page number (>= 1).")
        return
    if state["total_pages"] and num > state["total_pages"]:
        print(f"The document has only {state['total_pages']} pages.")
        return
    show_page(state, human_page=num)

