# GymNutriAI

Assistente de IA especializado em musculação, hipertrofia e nutrição esportiva baseado exclusivamente em evidências científicas. O sistema combina Retrieval-Augmented Generation (RAG) com fine-tuning de Small Language Model (SLM) para gerar respostas contextualizadas, com citações peer-reviewed e grading de evidência.

## Arquitetura

```
Usuario
   |
   v
FastAPI (REST API)
   |
   +---> RAG Pipeline
   |       Hybrid Search (BM25 + Semantic + RRF)
   |       +---> ChromaDB (vector store)
   |       +---> Cross-Encoder Reranker (bge-reranker-v2-m3)
   |
   +---> LLM Inference
           Phi-3.5-mini-instruct (3.8B params)
           + LoRA adapters (fine-tuned)
           Fallback: Groq API (llama-3.1-8b)
```

## Stack

- **LLM:** Microsoft Phi-3.5-mini-instruct (MIT License) — fine-tuned com Unsloth (QLoRA 4-bit) no Google Colab T4
- **RAG:** ChromaDB + sentence-transformers + BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranker
- **API:** FastAPI + Pydantic + Uvicorn
- **Deploy:** Render (free tier) + Groq API fallback

## Funcionalidades

- Respostas com citações no formato `[Autor Ano, Grau X]`
- Sistema de grading de evidência (A a E)
- Guardrails contra conteudo perigoso (insulina exogena, anabolizantes, SARMs)
- Documentacao automatica via Swagger UI (`/docs`)
- Fallback para Groq API quando o modelo local nao esta disponivel

## Endpoints

| Metodo | Rota | Descricao |
|--------|------|-----------|
| GET | `/` | Status da API |
| POST | `/chat` | Chat com RAG + LLM |
| POST | `/workout` | Gerar plano de treino estruturado |
| GET | `/health` | Health check |
| GET | `/docs` | Documentacao interativa (Swagger UI) |

## Como rodar localmente

```bash
# 1. Clone o repositorio
git clone https://github.com/Lorran207/gymnutriai.git
cd gymnutriai

# 2. Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instale as dependencias
pip install -r requirements.txt

# 4. Configure as variaveis de ambiente (opcional)
export MODEL_PATH="./lora_adapter"
export RAG_DB_PATH="./chroma_db"

# 5. Execute
uvicorn app:app --reload
```

Acesse `http://localhost:8000/docs` para a documentacao interativa.

## Variaveis de Ambiente

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `MODEL_PATH` | `./lora_adapter` | Caminho para os adapters LoRA |
| `RAG_DB_PATH` | `./chroma_db` | Caminho para o banco ChromaDB |
| `USE_GROQ_FALLBACK` | `false` | Ativar fallback para Groq API |
| `GROQ_API_KEY` | `""` | Chave da API Groq (obrigatoria se fallback ativo) |

## Dataset

O fine-tuning foi realizado com 40 pares de pergunta e resposta em formato ChatML/JSONL, cobrindo:

- Proteina, creatina, suplementacao
- Hipertrofia, volume, intensidade, RIR/RPE
- Tecnicas avancadas (rest-pause, myo-reps, cluster sets, BFR, DC Training, HIT)
- Jejum intermitente, insulina endogena, sensibilidade insulínica
- Sono, periodizacao, cardio
- Guardrails e seguranca

## Licencas

- **Codigo deste projeto:** MIT License (c) 2026 Lorran
- **Modelo de linguagem:** [Microsoft Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) — MIT License
- **Embeddings:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache 2.0
- **Reranker:** [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — MIT License

## Disclaimer

As informacoes fornecidas pelo GymNutriAI sao para fins educacionais e nao substituem orientacao medica ou de um nutricionista registrado. Nunca recomende insulina exogena, anabolizantes, SARMs ou medicamentos de prescricao.
