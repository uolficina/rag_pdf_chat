# RAG em PDF com Mistral

Script em Python que realiza busca de trechos em PDFs, reranqueia resultados e gera respostas usando o modelo de chat da Mistral. Foi pensado para rodar localmente via terminal, recebendo o caminho de um PDF e permitindo perguntas iterativas sobre o documento.

## Requisitos
- Python 3.10+ recomendado
- Chave da API da Mistral em `MISTRAL_API_KEY`
- Dependencias Python:
  - `pypdf`
  - `sentence_transformers`
  - `faiss-cpu`
  - `mistralai`
  - `numpy`

## Instalação rapida
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pypdf sentence_transformers faiss-cpu mistralai numpy
```

## Variaveis de ambiente
Defina a chave da Mistral antes de executar:
```bash
export MISTRAL_API_KEY="sua-chave-aqui"  # Windows: set MISTRAL_API_KEY=sua-chave-aqui
```

## Como usar
1) Coloque o PDF em um caminho acessivel (pode estar na mesma pasta).
2) Rode o script:
```bash
python rag.py
```
3) Informe o caminho do PDF quando solicitado.
4) Digite perguntas sobre o documento. Use `sair` para encerrar.

## O que o script faz
- **Leitura do PDF:** `carregar_pdf` extrai o texto pagina a pagina e registra offsets.
- **Divisao em trechos:** `divisor_trechos` cria janelas de texto (2000 chars com overlap 400) mantendo o indice da pagina.
- **Indexacao vetorial:** usa `SentenceTransformer` (`intfloat/multilingual-e5-base`) para embeddings normalizados e cria um indice FAISS para busca por similaridade.
- **Rerank:** busca top-30 no indice e reranqueia com `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) retornando os 3 melhores trechos.
- **Geração:** monta um prompt com os trechos escolhidos e envia para o modelo `mistral-small-latest` via SDK `mistralai`.
- **Loop de chat:** permite perguntas sucessivas usando o mesmo indice em memoria.

## Ajustes rapidos
- Tamanho dos trechos: altere `chunk_size` e `overlap` em `divisor_trechos`.
- Numero de candidatos e resultados finais: ajuste `k_base` e `k_final` em `busca_rerank`.
- Temperatura do modelo: mude `temperature` na funcao `mistral`.

## Problemas comuns
- **Chave nao configurada:** erro "Defina a API MISTRAL" indica `MISTRAL_API_KEY` ausente ou vazia.
- **PDF sem texto extraivel:** `pypdf` depende de texto incorporado; PDFs com apenas imagens exigem OCR previo.
- **Memoria/tempo:** PDFs grandes geram muitos chunks; reduza `chunk_size` ou limite paginas se notar lentidao.

## Exemplo rapido (Linux/macOS)
```bash
export MISTRAL_API_KEY="minha-chave"
python rag.py
# Digite o caminho: /caminho/para/documento.pdf
# Pergunta: Qual o resumo do capitulo 1?
```
