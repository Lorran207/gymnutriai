"""
GymNutriAI — API FastAPI para deploy no Render (Free Tier)
============================================================

Modos de operação (via variáveis de ambiente):
  - Render (recomendado): USE_GROQ_FALLBACK=true + GROQ_API_KEY
  - Local com GPU: lora_adapter/ + chroma_db/ na raiz do projeto

Para rodar localmente:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Para deploy no Render:
    Build Command: pip install -r requirements.txt
    Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os

import numpy as np

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_PATH = os.getenv("MODEL_PATH", "./lora_adapter")
RAG_DB_PATH = os.getenv("RAG_DB_PATH", "./chroma_db")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "./system_prompt_production.txt")
USE_GROQ_FALLBACK = os.getenv("USE_GROQ_FALLBACK", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# No Render free tier, pule o carregamento do LLM local (sem GPU/RAM)
SKIP_LOCAL_MODEL = os.getenv("SKIP_LOCAL_MODEL", "false").lower() == "true"
# RAG leve: desative se a instância não tiver RAM para sentence-transformers
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"

DEFAULT_SYSTEM_PROMPT = (
    "Você é o GymNutriAI, assistente especializado em treinamento de força, "
    "hipertrofia e nutrição esportiva. Baseie suas respostas em evidências "
    "científicas. Sempre inclua citações no formato [Autor Ano, Grau X]. "
    "NUNCA recomende insulina exógena, anabolizantes ou medicamentos. "
    "Responda em português brasileiro."
)


def load_system_prompt() -> str:
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = load_system_prompt()

# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Pergunta do usuário")
    use_rag: bool = Field(True, description="Usar RAG para enriquecer a resposta")
    max_tokens: int = Field(512, ge=50, le=2048, description="Máximo de tokens na resposta")
    temperature: float = Field(0.7, ge=0.1, le=1.0, description="Temperatura de geração")


class Source(BaseModel):
    id: str
    text: str
    source: str
    grade: str
    score: float


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Source] = []
    model_used: str
    disclaimer: str = (
        "Esta informação é para fins educacionais e não substitui orientação "
        "médica ou de um nutricionista registrado."
    )


class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    rir: int = Field(..., ge=0, le=5, description="Reps in Reserve")
    rest_seconds: int
    notes: Optional[str] = None
    evidence_grade: str = "B"


class WorkoutPlan(BaseModel):
    muscle_group: str
    exercises: List[Exercise]
    total_sets: int
    estimated_duration_min: int
    notes: str


# =============================================================================
# ESTADO GLOBAL
# =============================================================================

model = None
tokenizer = None
embedding_model = None
reranker = None
collection = None
bm25 = None
all_chunks: list = []
torch = None


def _load_local_llm():
    global model, tokenizer, torch

    if SKIP_LOCAL_MODEL:
        print("ℹ️ SKIP_LOCAL_MODEL=true — pulando carregamento do LLM local")
        return

    if not os.path.isdir(ADAPTER_PATH):
        print(f"⚠️ Adapter não encontrado em {ADAPTER_PATH}")
        return

    try:
        import torch as _torch
        from unsloth import FastLanguageModel

        torch = _torch
        print(f"🔄 Carregando base {BASE_MODEL} + adapter {ADAPTER_PATH}...")

        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        FastLanguageModel.for_inference(model)
        print("✅ Modelo Phi-3.5-mini + LoRA carregado")
    except ImportError:
        print("⚠️ unsloth/peft não instalados — use requirements-colab.txt ou Groq")
    except Exception as e:
        print(f"⚠️ Erro ao carregar modelo local: {e}")
        model = None


def _load_rag():
    global embedding_model, reranker, collection, bm25, all_chunks

    if not ENABLE_RAG:
        print("ℹ️ ENABLE_RAG=false — RAG desativado")
        return

    if not os.path.exists(RAG_DB_PATH):
        print(f"⚠️ ChromaDB não encontrado em {RAG_DB_PATH}")
        return

    try:
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        client = chromadb.PersistentClient(path=RAG_DB_PATH)
        collection = client.get_or_create_collection(name="gymnutriai_docs")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        docs = collection.get(include=["documents", "metadatas"])
        if not docs["documents"]:
            print("⚠️ ChromaDB vazio")
            return

        all_chunks = []
        for chunk_id, doc, meta in zip(
            docs["ids"],
            docs["documents"],
            docs["metadatas"],
        ):
            all_chunks.append({
                "id": chunk_id,
                "text": doc,
                "source": meta.get("source", "Desconhecido"),
                "grade": meta.get("grade", "C"),
                "doc_id": meta.get("doc_id", chunk_id),
            })

        corpus = [c["text"] for c in all_chunks]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ RAG carregado: {len(all_chunks)} chunks")
    except ImportError:
        print("⚠️ Dependências RAG não instaladas (chromadb, sentence-transformers)")
    except Exception as e:
        print(f"⚠️ Erro ao carregar RAG: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Iniciando GymNutriAI...")
    _load_local_llm()
    _load_rag()
    yield
    print("👋 Encerrando GymNutriAI")


app = FastAPI(
    title="GymNutriAI API",
    description=(
        "Assistente de IA especializado em musculação e nutrição esportiva "
        "baseado em evidências científicas."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# RETRIEVAL
# =============================================================================


def retrieve_documents(query: str, top_k: int = 3) -> List[Source]:
    if not all([collection, bm25, all_chunks, embedding_model, reranker]):
        return []

    query_emb = embedding_model.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(10, len(all_chunks)),
        include=["documents", "metadatas", "distances"],
    )

    semantic_scores = {}
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        semantic_results["ids"][0],
        semantic_results["documents"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0],
    )):
        score = 1.0 / (1.0 + dist)
        semantic_scores[doc_id] = {
            "text": doc,
            "source": meta.get("source", "Desconhecido"),
            "grade": meta.get("grade", "C"),
            "semantic_rank": i + 1,
        }

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:10]

    bm25_results = {}
    for rank, idx in enumerate(top_bm25_idx, 1):
        if bm25_scores[idx] > 0:
            chunk = all_chunks[idx]
            bm25_results[chunk["id"]] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "grade": chunk["grade"],
                "bm25_rank": rank,
            }

    k_rrf = 60
    fused_scores = {}
    all_ids = set(semantic_scores.keys()) | set(bm25_results.keys())

    for doc_id in all_ids:
        score = 0.0
        if doc_id in semantic_scores:
            score += 1.0 / (k_rrf + semantic_scores[doc_id]["semantic_rank"])
        if doc_id in bm25_results:
            score += 1.0 / (k_rrf + bm25_results[doc_id]["bm25_rank"])

        if doc_id in semantic_scores:
            entry = semantic_scores[doc_id]
        else:
            entry = bm25_results[doc_id]

        fused_scores[doc_id] = {
            "text": entry["text"],
            "source": entry["source"],
            "grade": entry["grade"],
            "rrf_score": score,
        }

    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True,
    )
    top_rrf = sorted_docs[: top_k * 2]

    if not top_rrf:
        return []

    pairs = [[query, doc[1]["text"]] for doc in top_rrf]
    rerank_scores = reranker.predict(pairs)

    reranked = []
    for i, (doc_id, doc_data) in enumerate(top_rrf):
        reranked.append(Source(
            id=doc_id,
            text=doc_data["text"],
            source=doc_data["source"],
            grade=doc_data["grade"],
            score=float(rerank_scores[i]),
        ))

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k]


# =============================================================================
# GERAÇÃO
# =============================================================================


def _decode_assistant_response(inputs, outputs) -> str:
    input_length = inputs.shape[-1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response.strip()


def generate_with_local_model(
    query: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if model is None or tokenizer is None:
        return None

    if context:
        user_content = (
            f"Use o seguinte contexto científico para responder:\n\n"
            f"### CONTEXTO ###\n{context}\n\n### PERGUNTA ###\n{query}"
        )
    else:
        user_content = query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if torch is not None and torch.cuda.is_available():
        inputs = inputs.to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return _decode_assistant_response(inputs, outputs)


def generate_with_groq(
    query: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    import requests

    if not GROQ_API_KEY:
        return None

    user_content = query
    if context:
        user_content = f"Contexto científico:\n{context}\n\nPergunta: {query}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ Erro Groq: {e}")
        return None


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    return {
        "message": "GymNutriAI API está online!",
        "version": "1.1.0",
        "model_loaded": model is not None,
        "rag_loaded": collection is not None,
        "groq_configured": bool(GROQ_API_KEY),
        "docs_indexed": len(all_chunks),
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    sources: List[Source] = []
    context = ""
    if request.use_rag:
        sources = retrieve_documents(request.query)
        if sources:
            context = "\n\n".join(
                f"[{s.source}, Grau {s.grade}] {s.text}" for s in sources
            )

    response_text = None
    model_used = "unknown"

    if model is not None:
        response_text = generate_with_local_model(
            request.query, context, request.max_tokens, request.temperature
        )
        if response_text:
            model_used = "phi-3.5-mini-qlora"

    if response_text is None and USE_GROQ_FALLBACK:
        response_text = generate_with_groq(
            request.query, context, request.max_tokens, request.temperature
        )
        if response_text:
            model_used = "groq-llama-3.1-8b"

    if response_text is None:
        detail = "Modelo não disponível."
        if USE_GROQ_FALLBACK and not GROQ_API_KEY:
            detail += " Configure GROQ_API_KEY no Render."
        elif not USE_GROQ_FALLBACK and model is None:
            detail += " Ative USE_GROQ_FALLBACK=true ou forneça lora_adapter/."
        raise HTTPException(status_code=503, detail=detail)

    return QueryResponse(
        query=request.query,
        response=response_text,
        sources=sources,
        model_used=model_used,
    )


@app.post("/workout", response_model=WorkoutPlan)
async def generate_workout(
    muscle_group: str,
    level: str = "intermediario",
    style: str = "moderado",
    days_per_week: int = 1,
):
    plans = {
        "peito": WorkoutPlan(
            muscle_group="Peitoral",
            exercises=[
                Exercise(
                    name="Supino reto com barra",
                    sets=3,
                    reps="6-8",
                    rir=2,
                    rest_seconds=180,
                    evidence_grade="A",
                ),
                Exercise(
                    name="Supino inclinado com halteres",
                    sets=3,
                    reps="8-10",
                    rir=1,
                    rest_seconds=120,
                    evidence_grade="B",
                ),
                Exercise(
                    name="Crucifixo na máquina",
                    sets=2,
                    reps="10-12",
                    rir=1,
                    rest_seconds=90,
                    evidence_grade="B",
                ),
            ],
            total_sets=8,
            estimated_duration_min=45,
            notes=(
                "Foco em tensão mecânica e proximidade da falha controlada. "
                "Ajuste volume conforme recuperação."
            ),
        ),
        "costas": WorkoutPlan(
            muscle_group="Costas",
            exercises=[
                Exercise(
                    name="Levantamento terra",
                    sets=3,
                    reps="5-6",
                    rir=2,
                    rest_seconds=240,
                    evidence_grade="A",
                ),
                Exercise(
                    name="Remada curvada",
                    sets=3,
                    reps="8-10",
                    rir=2,
                    rest_seconds=150,
                    evidence_grade="B",
                ),
                Exercise(
                    name="Puxada aberta",
                    sets=3,
                    reps="10-12",
                    rir=1,
                    rest_seconds=120,
                    evidence_grade="B",
                ),
            ],
            total_sets=9,
            estimated_duration_min=50,
            notes=(
                "Priorize compostos pesados. O terra é opcional se houver "
                "limitação de mobilidade."
            ),
        ),
    }

    key = muscle_group.lower()
    if key not in plans:
        raise HTTPException(
            status_code=404,
            detail=f"Grupo '{muscle_group}' não encontrado. Disponíveis: {list(plans.keys())}",
        )
    return plans[key]


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch is not None and torch.cuda.is_available(),
        "model_loaded": model is not None,
        "rag_available": collection is not None,
        "groq_configured": bool(GROQ_API_KEY),
    }
