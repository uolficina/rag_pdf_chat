import os  # utilitários de caminho/arquivo
from pypdf import PdfReader  # leitor de PDFs
from sentence_transformers import SentenceTransformer, CrossEncoder  # embeddings e reranker
import numpy as np  # operações numéricas
import faiss  # índice vetorial
from mistralai import Mistral  # cliente Mistral


### CARREGADOR DE PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)  # abre o PDF
    texts = []  # guarda o texto de cada página
    offsets = []  # guarda o índice inicial de cada página no texto completo
    cursor = 0  # posição acumulada de caracteres
    for page in reader.pages:  # percorre as páginas
        page_text = page.extract_text() or ""  # extrai texto da página
        offsets.append(cursor)  # registra onde a página começa
        texts.append(page_text)  # adiciona o texto desta página
        cursor += len(page_text) + 1  # avança cursor (+1 pelo "\n" na junção)
    full_text = "\n".join(texts)  # concatena páginas com quebras de linha
    return full_text, offsets  # devolve texto completo e offsets por página

###SEPARADOR DE PAGINAS
def find_page(start_idx, page_offsets):
    page = 0  # índice da página (0-based)
    for i, off in enumerate(page_offsets):  # percorre offsets
        if start_idx < off:  # se passou do início da próxima página
            break  # sai; a página atual é a anterior
        page = i  # atualiza página corrente
    return page  # retorna índice da página encontrada

###DIVISOR DE TRECHOS
def split_text(text, page_offsets, chunk_size=1000, overlap=200):
    if chunk_size <= overlap:  # valida limites de chunk/overlap
        raise ValueError("documento menor que o chunk minimo")  # erro se inválido
    chunks = []  # lista que receberá os pedaços
    step = chunk_size - overlap  # passo de avanço com overlap
    for start in range(0, len(text), step):  # percorre o texto em passos fixos
        end = start + chunk_size  # calcula fim do chunk
        chunk_text = text[start:end]  # extrai fatia do texto
        page = find_page(start, page_offsets)  # descobre página de origem
        chunks.append({"page": page, "text": chunk_text})  # guarda página e texto
    return chunks  # devolve todos os chunks

##CHAMADA  PRINCIPAL
def main():
    file_path = input("Digite o caminho do pdf: ").strip().strip('"')  # lê caminho do PDF
    if not file_path:  # nenhum caminho fornecido
        print("Nenhum arquivo informado")  # avisa falta de entrada
        return  # encerra
    if not os.path.isfile(file_path):  # verifica se o arquivo existe
        print(f"Arquivo não encontrado: {file_path}")  # avisa ausência
        return  # encerra
### CARREGA OS TRECHOS REFERENCIANDO PAGINAS
    pdf_text, offsets = load_pdf(file_path)  # carrega texto completo e offsets
    print("Trechos carregados")  # confirma carregamento
    #print(pdf_text[:1000])  # mostra os primeiros 1000 caracteres
    chunks = split_text(pdf_text, offsets, chunk_size=1000, overlap=200)  # fatia texto em chunks
    chunk_texts = [c["text"] for c in chunks]  # extrai somente o texto de cada chunk para embeddar
### CARREGA MODELO DE EMBENDINGS TRANSFORMERS PARA BUSCAR TRECHOS EM RAG
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # carrega modelo de embeddings
    embs = embed_model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")  # gera vetores normalizados
    print("Modelo SentenceTransformer carregado")
### CRIA INDICE FAISS
    d = embs.shape[1]  # dimensão dos vetores
    index = faiss.IndexFlatIP(d)  # índice FAISS de produto interno (para cosseno com vetores normalizados)
    index.add(embs)  # adiciona todos os vetores ao índice
    print("Índice FAISS criado")
### CARREGAD CROSSENCODER MARCO PARA RANKEAR O QUE O MODELO CAPTOU COM O INDICE
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # reranker mais preciso (cross-encoder)
    print("Modelo CrossEncoder Carregado")

# FUNÇÃO DE RERANK COM  INDICE E CROSS ENCODER
    def buscar(pergunta, k_base=30, k_final=3):  # busca no FAISS e reranqueia, retornando top k_final
        q_emb = embed_model.encode([pergunta], convert_to_numpy=True, normalize_embeddings=True).astype("float32")  # embedding da pergunta
        scores, idxs = index.search(q_emb, k_base)  # traz k_base candidatos aproximados

        pares = [(pergunta, chunk_texts[i]) for i in idxs[0]]  # monta pares pergunta-contexto para reranking
        rerank_scores = cross.predict(pares)  # score mais preciso do cross-encoder
        reranked = sorted(zip(rerank_scores, idxs[0]), key=lambda x: x[0], reverse=True)[:k_final]  # ordena por score e pega top k_final
        resultados = []  # lista final de chunks selecionados
        for score, i in reranked:
            resultados.append({
                "score": float(score),  # score do cross-encoder
                "page": chunks[i]["page"],  # página de origem
                "text": chunks[i]["text"],  # texto do chunk
            })
        return  resultados  # devolve chunks reranqueados

    while True:
        pergunta = input("Digite sua pergunta (ou 'sair'): ").strip()  # lê a pergunta do usuário
        if pergunta.lower() == "sair":  # condição de saída
            break  # encerra loop
        resultados = buscar(pergunta)  # obtém melhores chunks para a pergunta
        #print("Top resultados:")  # cabeçalho informativo
        for r in resultados:  # percorre chunks selecionados
            print(f"\nScore: {r['score']:.4f} | Página: {r['page']}")  # mostra score e página
            print(r["text"][:500])  # mostra parte do texto do chunk

        # chama o LLM com os chunks selecionados para responder à pergunta
        resposta, uso = mistral(pergunta, resultados)  # envia pergunta + contexto ao Mistral
        print("\nResposta (Mistral):\n")  # separador visual
        print(resposta)  # mostra a resposta gerada
        print(
            f"\nTokens - entrada: {uso.prompt_tokens}, saída: {uso.completion_tokens}, total: {uso.total_tokens}"
        )
### BLOCO MISTRAL CHAMA MODELO E  DEFINE PROMPT ALIMENTADO COM A SAIDA DAS FUNÇÕES ANTERIORES
mistral_model = "mistral-small-latest"

def prompt(pergunta, contextos):
    blocos = []  # acumula trechos formatados com página
    for ctx in contextos:  # percorre cada contexto selecionado
        blocos.append(f"[Página {ctx['page']}] {ctx['text']}")  # anota página e texto
    contexto = "\n\n".join(blocos)  # junta todos os trechos separados por linha em branco
    return (
        "Resuma ou responda à pergunta usando apenas o contexto fornecido. "  # instrução ao modelo
        "Se não houver resposta, diga que não sabe.\n\n"  # reforça para não inventar
        f"Pergunta: {pergunta}\n\n"  # insere pergunta do usuário
        f"Contexto:\n{contexto}"  # insere trechos do PDF
    )

def mistral(pergunta, contextos):
    api_key = "9BA3otp3UZQ5LYZP6AAaYeUHlMeVshey"  # chave da API (hardcoded; não seguro)
    if not api_key:  # valida se chave está definida
        raise RuntimeError("Defina a API MISTRAL")  # alerta para configurar
    client = Mistral(api_key=api_key)  # instancia cliente Mistral
    print("Modelo Mistral Carregado")
    prompt_text = prompt(pergunta, contextos)  # monta prompt com pergunta e contextos
    resp = client.chat.complete(
        model=mistral_model,  # modelo escolhido
        messages=[{"role": "user", "content": prompt_text}],  # mensagem do usuário
        temperature=0.2,  # baixa temperatura para respostas mais fiéis ao contexto
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage  # devolve texto da resposta

if __name__ == "__main__":  # executa main se rodar diretamente
    main()  # chama fluxo principal
