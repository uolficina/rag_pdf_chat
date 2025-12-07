import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from mistralai import Mistral
import textwrap
import time 

# variaveis LLM
modelo_mistral = "mistral-small-latest"
modelo_embedings = "intfloat/multilingual-e5-base"
modelo_crossencoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#api
mistral_api = os.getenv("MISTRAL_API_KEY")
total_paginas = 0  # será preenchido ao carregar o PDF
paginas_texto = []  # texto bruto de cada página
embed_model = None
index = None
cross = None
chunks = []
chunk_texts = []
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
    return full_text, offsets, texts #devolve texto completo, offsets e textos por pagina

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
    global total_paginas, paginas_texto
    pdf_text, offsets, paginas_texto = carregar_pdf(file_path) #carrega texto completo
    total_paginas = len(offsets)
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

def prompt_mistral(pergunta, contextos):
    blocos = []  # acumula trechos formatados com página
    for ctx in contextos:  # percorre cada contexto selecionado
        blocos.append(f"[Página {ctx['page']}] {ctx['text']}")  # anota página e texto
    contexto = "\n\n".join(blocos)  # junta todos os trechos separados por linha em branco
    return (
        "Resuma ou responda à pergunta usando apenas o contexto fornecido. "  # instrução ao modelo
        "Se não houver resposta, diga que não sabe.\n\n"
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

def gerar_titulo_chunk(trecho, tentativas=5, base_delay=2):
    api_key = mistral_api
    client = Mistral(api_key=api_key)

    prompt = (
        "Gere APENAS UM TITULO CURTO para o texto abaixo.\n"
        "Regras do titulo:\n"
        "- Deve ter entre 2 a 5 palavras.\n"
        "- Não descreva frases longas.\n"
        "- Não escreva explicações.\n"
        "- Não use dois-pontos.\n\n"
        f"Trecho:\n{trecho}"
    )

    for tentativa in range(1, tentativas +1):
        try: 
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            status = getattr(e, "http_status", None) or getattr(e, "status_code", None)
            if (status == 429) or ("rate limit" in msg) or ("too many requests" in msg):
                if tentativa == tentativas:
                    raise
                espera = base_delay * (2 ** (tentativa - 1))
                print(f"Rate limit; aguardando  {espera:.1f}s antes de tentar novamente...")
                time.sleep(espera)
                continue
            raise

def gerar_indice(chunks):
    paginas_exibidas = set()
    for c in chunks:
        pagina = c["page"]
        if pagina in paginas_exibidas:
            continue
        paginas_exibidas.add(pagina)
        trecho = c["text"]
        titulo = gerar_titulo_chunk(trecho)
        print(f"Página {pagina}: {titulo}")

def formatar_texto(texto, largura=80):
    linhas = []
    for paragrafo in texto.splitlines():
        for linha in textwrap.wrap(paragrafo, largura):
            if len(linha) < largura - len(linha):
               deficit = largura - len(linha)
               gaps = linha.count("")
               partes = linha.split("")
               base, resto = divmod(deficit, gaps)
               for i in range(resto):
                   partes[i] += " "
               linha = (" " * base).join(partes)
            linhas.append(linha)
    return "\n".join(linhas)

def exibir_pagina(chunks, pagina_humana, trim=None):
    pagina_idx = pagina_humana - 1  # alinha contagem humana (1…) com índice interno (0…)
    if total_paginas and (pagina_idx < 0 or pagina_idx >= total_paginas):
        print(f"O documento tem apenas {total_paginas} páginas.")
        return
    if paginas_texto and 0 <= pagina_idx < len(paginas_texto):
        texto = paginas_texto[pagina_idx]
        if trim and len(texto) > trim:
            texto = texto[:trim].rstrip() + "..."
        print(f"\nPágina {pagina_humana} (texto completo):\n{texto}")
        return
    trechos = [c for c in chunks if c["page"] == pagina_idx]
    if not trechos:
        print(f"Nenhum trecho encontrado na página {pagina_humana}")
        return
    for i, c in enumerate(trechos, 1):
        texto = c["text"]
        texto = formatar_texto(texto, largura=80)
        if trim and len(texto) > trim:
            texto = texto[:trim].rstrip() + "..."
        print(f"\nTrecho {i} (página {pagina_humana}):\n{texto}")
       
def escolher_pagina():
    pagina_escolhida = input("Digite o número da página que você quer ler (ex: 1, 2, 3...): ")
    try:
        num = int(pagina_escolhida)
        if num <= 0:
            raise ValueError
    except ValueError:
        print("Informe um número de página válido (>= 1).")
        return
    if total_paginas and num > total_paginas:
        print(f"O documento tem apenas {total_paginas} páginas.")
        return
    exibir_pagina(chunks, pagina_humana=num)   


def chat():
    while True:
        print("\n===Menu de Funções ===")
        print("1) Carregar PDF")
        print("2) Conversar com o PDF")
        print("3) Gerar Índice Semântico")
        print("4) Mostrar página específica")
        print("5) Sair")

        opção = input("Escolha um número do menu: ").strip()

        if opção == "1":
            file_path = input("Digite o caminho do PDF ou digite 'voltar': ").strip().strip('"')
            if file_path == "voltar":
                return chat()
            if not file_path:
                print("Nenhum arquivo informado.")
                return
            if not os.path.isfile(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                return
            global embed_model,index, cross, chunks, chunk_texts
            embed_model,index,cross,chunks, chunk_texts = rag(file_path)
        elif opção == "2":
            if not chunks:
                print("Carregue um pdf primeiro")
                return chat()
            while True:
                pergunta= input("Digite sua pergunta ou 'voltar':  ").strip().lower()
                if pergunta == "voltar":
                    return chat()
                resultados  = busca_rerank(pergunta)
                resposta, uso = mistral(pergunta,resultados)
                print("\nRESPOSTA (Mistral):\n")
                print(resposta)
                print(f"\nTokens - entrada: {uso.prompt_tokens}, saída: {uso.completion_tokens}, total: {uso.total_tokens}")
        elif opção == "3":
            if not chunks:
                print("Carregue um pdf primeiro")
                continue
            gerar_indice(chunks)
        elif opção == "4":
            if not chunks:
                print("Carregue um pdf primeiro")
                continue
            escolher_pagina()
        elif opção == "5":
            break
        
if __name__ == "__main__":  # executa main se rodar diretamente
    chat()  # chama fluxo principal        
