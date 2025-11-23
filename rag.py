import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from mistralai import Mistral
# variaveis LLM
modelo_mistral = "mistral-small-latest"
modelo_embedings = "intfloat/multilingual-e5-base"
modelo_crossencoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#api
mistral_api = os.getenv("MISTRAL_API_KEY")

#funções
def carregar_pdf(file_path):
    reader = PdfReader(file_path) #abre o pdf
    texts =  [] #guarda o texto de cada pagina
    offsets = [] # guarda o indice inicial da pagina
    cursor = 0 #posicao acumulada de caracteres
    for page in reader.pages: #percorre paginas
        page_text = page.extract_text() or "" #extrai texto da pagina
        offsets.append(cursor) #registra onde a pagina começa
        texts.append(page_text) #adicona o texto desta pagina
        cursor += len(page_text) + 1 #avança o cursos +1 na quebra de linha
    full_text = "\n".join(texts) # concatena paginas com quebras de linha
    return full_text, offsets #devolve texto completo e offsets por pagina

def encontra_paginas(start_idx, page_offsets):
    page = 0 #indice da pagina
    for i, off in enumerate(page_offsets): #percorre offsets
        if start_idx < off: # verifica se passou do inicio da proxima pagina
            break
        page = i #atualiza a pagina atual
    return page #retorna o indice da pagina encontrada

def divisor_trechos(text, page_offsets, chunk_size=2000, overlap=400):
    if chunk_size <= overlap: #valida limites de chunk overlap
        raise ValueError("Documento menor que o trecho minimo permitido") 
    chunks = [] #lista que vai receber os trechos
    step = chunk_size - overlap #avanço para overlap
    for start in range(0, len(text), step): #percorre o texto
        end = start + chunk_size #cacula o fim do trecho
        chunk_text = text[start:end] # extrai parte do texto
        page = encontra_paginas(start, page_offsets) #descobre a pagina de origem do trecho
        chunks.append({"page": page, "text": chunk_text}) #guarda a pagina junto com otrecho
    return chunks #devolve trechos
    print("PDF COM TRECHOS DEMARCADOS")

def rag(file_path):
    pdf_text, offsets = carregar_pdf(file_path) #carrega texto completo
    chunks = divisor_trechos(pdf_text, offsets, chunk_size=2000, overlap=400)
    chunk_texts = [c["text"] for c in chunks]
    embed_model = SentenceTransformer(modelo_embedings)
    print("MODELO DE EMBEDINGS CARREGADO")
    embed_texts = [f"passage: {txt}" for txt in chunk_texts]
    embs = embed_model.encode(embed_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    d = embs.shape[1] #dimensao dos vetores  para o faiss
    index = faiss.IndexFlatIP(d) #indice faiss criado
    index.add(embs) #adiciona todos os vetores ao indice faiss
    print("INDICE FAISS CRIADO")
    cross = CrossEncoder(modelo_crossencoder)
    print("CROSSENCODER CARREGADO")
    return embed_model, index, cross, chunks, chunk_texts

def busca_rerank(pergunta, k_base=30, k_final=3): #busca no faiss e rank top 3 entre os top 30
    global embed_model, index, cross, chunks, chunk_texts
    resultados = []
    pergunta_fmt = f"query: {pergunta}" #espera do e5 pela query
    q_emb = embed_model.encode([pergunta_fmt], convert_to_numpy=True, normalize_embeddings=True).astype("float32") #embedding da pergunta
    scores, idxs = index.search(q_emb, k_base) #traz candidatos
    pares = [(pergunta, chunk_texts[i]) for i in idxs[0]] #monta  pares de perguntas para rerank
    rerank_scores = cross.predict(pares) #score mais preciso do modelo crossencoder
    reranked = sorted(zip(rerank_scores, idxs[0]), key=lambda x: x[0], reverse=True)[:k_final] #ordena por score
    for score, i in reranked:
        resultados.append({
            "score": float(score), #score do crossencoder
            "page": chunks[i]["page"], #pagina de origem
            "text": chunks[i]["text"], #texto do trecho
        })
        return resultados #devolve trechos rankeados
        print("RANKING DE TRECHOS ELABORADO")

def prompt_mistral(pergunta, contextos):
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
    api_key = mistral_api
    if not api_key:  # valida se chave está definida
        raise RuntimeError("Defina a API MISTRAL")  # alerta para configurar
    client = Mistral(api_key=api_key)  # instancia cliente Mistral
    print("MODELO MISTRAL CARREGADO")
    prompt_text = prompt_mistral(pergunta, contextos)  # monta prompt com pergunta e contextos
    resp = client.chat.complete(
        model=modelo_mistral,  # modelo escolhido
        messages=[{"role": "user", "content": prompt_text}],  # mensagem do usuário
        temperature=0.2,  # baixa temperatura para respostas mais fiéis ao contexto
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage  # devolve texto da resposta

def chat():
    file_path = input("Digite o caminho do pdf: ").strip().strip('"') #lê o caminho do pdf
    if not file_path: #se nenhum caminho fornecido
        print("Nenhum arquivo informado") #avisa a falta de entrada
        return #encerra
    if not os.path.isfile(file_path): #verifica se existe o arquivo informado
        print(f"Arquivo não encontrado: {file_path}") #avisa que o arquivo não foi encontrado
        return #encerra
    global embed_model, index, cross, chunks, chunk_texts
    embed_model, index, cross, chunks, chunk_texts = rag(file_path)
    while True:
        pergunta= input("Digite sua pergunta ao documento (ou digite 'sair' para encerrar): ").strip() #le a pergunta do usuario
        if pergunta.lower() == "sair": #condição  de saida
            break #encerra 
        resultados = busca_rerank(pergunta) #obtem retorno da função busca rerank com os melhores trechos
        #for r in resultados:  # percorre chunks selecionados
        #        print(f"\nScore: {r['score']:.4f} | Página: {r['page']}")  # mostra score e página
        #        print(r["text"][:500])  # mostra parte do texto do chunk
        resposta, uso = mistral(pergunta,resultados) #envia pergunta  e contexto  para mistral
        print("\nRESPOSTA (Mistral):\n")  # separador visual
        print(resposta)  # mostra a resposta gerada
        print(f"\nTokens - entrada: {uso.prompt_tokens}, saída: {uso.completion_tokens}, total: {uso.total_tokens}")

if __name__ == "__main__":  # executa main se rodar diretamente
    chat()  # chama fluxo principal        



