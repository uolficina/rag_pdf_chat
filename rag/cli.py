import os

from rag.config import state
from rag.index_store import list_docs, load_doc
from rag.pdf_utils import choose_page
from rag.retrieval import (
    generate_index,
    mistral_chat,
    prepare_document,
    search_rerank,
)


def choose_language():
    current = state.get("answer_lang") or "(automático)"
    print(f"\nIdioma atual da resposta: {current}")
    print("1) Português (PT-BR)")
    print("2) English")
    print("3) Automático (seguir prompt/padrão)")
    choice = input("Escolha uma opção: ").strip()
    if choice == "1":
        state["answer_lang"] = "pt"
        print("Idioma definido para português.")
    elif choice == "2":
        state["answer_lang"] = "en"
        print("Idioma definido para inglês.")
    elif choice == "3":
        state["answer_lang"] = ""
        print("Idioma resetado para automático.")
    else:
        print("Opção inválida; idioma não alterado.")


def chat_menu():
    while True:
        print("\n=== Function Menu ===")
        print("1) Load PDF")
        print("2) Chat with the PDF")
        print("3) Generate Semantic Index")
        print("4) Show specific page")
        print("5) Choose a loaded file from list")
        print("6) Choose answer language")
        print("7) Exit")

        option = input("Choose a menu number: ").strip()

        if option == "1":
            file_path = input("Enter the PDF path or type 'back': ").strip().strip('"')
            if file_path == "back":
                continue
            if not file_path:
                print("No file provided.")
                continue
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue
            prepare_document(file_path)
        elif option == "2":
            if not state["chunks"]:
                print("Load a PDF first")
                continue
            while True:
                question = input("Type your question or 'back':  ").strip()
                if question == "back":
                    break
                results = search_rerank(question)
                answer, usage = mistral_chat(question, results, preferred_lang=state.get("answer_lang"))
                print("\nANSWER (Mistral):\n")
                print(answer)
                print(f"\nTokens - input: {usage.prompt_tokens}, output: {usage.completion_tokens}, total: {usage.total_tokens}")
        elif option == "3":
            if not state["chunks"]:
                print("Load a PDF first")
                continue
            generate_index(state["chunks"])
        elif option == "4":
            if not state["chunks"]:
                print("Load a PDF first")
                continue
            choose_page(state)
        elif option == "5":
            docs = list_docs()
            if not docs:
                print("Theres no index saved yet")
                continue
            print("\n=== Index File List ===")
            for i, d in enumerate(docs, start=1):
                print(f"{i}) {d['name']} [{d['docid'][:8]}] pages: {d.get('total_pages', '?')}")

            choice = input("Choose a number or 'back' for the previous menu: ").strip()
            if choice == "back":
                continue
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(docs):
                    raise ValueError
            except ValueError:
                print("Invalid Choice")
                continue

            selected = docs[idx]
            load_doc(selected["docid"])
            print(f"Loaded: {selected['name']}")
        elif option == "6":
            choose_language()
        elif option == "7":
            break


if __name__ == "__main__":
    chat_menu()
