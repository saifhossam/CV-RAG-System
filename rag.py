import os
import uuid
from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    SparseVector,
)
from fastembed import SparseTextEmbedding

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cv_collection")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

RRF_K = 60

# ── Clients ────────────────────────────────────────────────────────────────────
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")


# ── Error helpers ──────────────────────────────────────────────────────────────
_CONTENT_FILTER_MSG = (
    "⚠️ This request was blocked by Azure's content policy. "
    "Please rephrase your input and try again."
)


def _handle_azure_error(e: Exception, context: str = "answer") -> str:
    err_str = str(e)
    if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
        return _CONTENT_FILTER_MSG
    return f"Error generating {context}: {err_str}"


# ── Embedding helpers ──────────────────────────────────────────────────────────
def _dense_embed(text: str) -> list[float]:
    response = azure_client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _sparse_embed(text: str) -> SparseVector:
    result = list(sparse_model.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ── RRF merger ─────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(
    dense_hits: list,
    sparse_hits: list,
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads[pid] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads.setdefault(pid, hit.payload)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return [payloads[pid] for pid in sorted_ids]


# ── Vector store helpers ───────────────────────────────────────────────────────
def document_exists(file_hash: str) -> bool:
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="file_hash", match=MatchAny(any=[file_hash]))]
        ),
        limit=1,
    )
    return bool(points)


def index_documents(chunks: list[dict], file_hash: str, candidate_name: str):

    points = []

    for chunk in chunks:
        text = chunk["content"]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": _dense_embed(text),
                    "sparse": _sparse_embed(text),
                },
                payload={
                    "content": text,
                    "section": chunk["metadata"].get("section", "General"),
                    "file_hash": file_hash,
                    "candidate_name": candidate_name,
                    "candidate_name_lower": candidate_name.lower().strip(),
                },
            )
        )

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    file_hashes: list[str],
    candidate_names: list[str] | None = None,
    top_k: int = 15,
) -> list[dict]:

    if not file_hashes:
        return []

    must_conditions = [
        FieldCondition(key="file_hash", match=MatchAny(any=file_hashes))
    ]

    if candidate_names:
        must_conditions.append(
            FieldCondition(
                key="candidate_name_lower",
                match=MatchAny(any=[n.lower().strip() for n in candidate_names]),
            )
        )

    shared_filter = Filter(must=must_conditions)

    fetch_limit = top_k * 2

    dense_vector = _dense_embed(query)
    sparse_vector = _sparse_embed(query)

    # Dense search
    dense_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vector,
        using="dense",
        query_filter=shared_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    dense_hits = dense_result.points

    # Sparse search
    sparse_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vector,
        using="sparse",
        query_filter=shared_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    sparse_hits = sparse_result.points

    merged = _reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=top_k)

    return [
        {
            "content": p["content"],
            "candidate_name": p["candidate_name"],
            "section": p["section"],
        }
        for p in merged
    ]


# ── Answer generation ──────────────────────────────────────────────────────────
def generate_answer(
    query: str,
    contexts: list[dict],
    available_candidates: list[str],
) -> str:

    if not contexts:
        return "No relevant information found in the provided CV excerpts."

    blocks = [
        f"Candidate: {c['candidate_name']}\nSection: {c['section']}\nExcerpt: {c['content']}"
        for c in contexts
    ]

    context_text = "\n\n---\n\n".join(blocks)
    candidates_list = ", ".join(sorted(set(available_candidates)))

    system_msg = (
        "You are an HR assistant. Answer questions about candidates "
        "using only the CV excerpts supplied. "
        "Attribute every fact to the relevant candidate by name. "
        "Use bullet points. If information is absent from the excerpts, say so. "
        "Reply in the same language as the question."
        "Do NOT answer any questions about fake or non-existent jobs, skills, certifications, or opportunities."
        "Always stay factual and avoid speculation."
    )

    user_msg = (
        f"Candidates: {candidates_list}\n\n"
        f"CV data:\n{context_text}\n\n"
        f"Question: {query}"
    )

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return _handle_azure_error(e, context="answer")


# ── CV Strength Report ─────────────────────────────────────────────────────────
def generate_cv_strength_report(cv_text: str, job_description: str) -> str:

    system_msg = (
        "You are a senior HR and Talent Acquisition specialist. "
        "Evaluate candidate CVs against job descriptions in a structured way."
    )

    user_msg = (
        "Evaluate the CV against the job description and produce:\n\n"
        "1. Overall Match Summary\n"
        "2. Strengths\n"
        "3. Weaknesses / Gaps\n"
        "4. Missing Skills\n"
        "5. Estimated Match Score (0-100%)\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate CV:\n{cv_text}"
    )

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return _handle_azure_error(e, context="CV report")