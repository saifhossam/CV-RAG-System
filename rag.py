import os
import uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)
from groq import Groq

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cv_collection")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# ✅ Updated to a current, supported Groq model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Clients ────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ── Vector store helpers ───────────────────────────────────────────────────────
def document_exists(file_hash: str) -> bool:
    """Return True if this file hash is already indexed in Qdrant."""
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="file_hash", match=MatchAny(any=[file_hash]))]
        ),
        limit=1,
    )
    return bool(points)


def index_documents(chunks: list[dict], file_hash: str, candidate_name: str):
    """Encode and upsert CV chunks into Qdrant."""
    points = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk["content"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content": chunk["content"],
                    "section": chunk["metadata"].get("section", "General"),
                    "file_hash": file_hash,
                    "candidate_name": candidate_name,
                    # lowercase for case-insensitive candidate name filtering
                    "candidate_name_lower": candidate_name.lower().strip(),
                },
            )
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(
    query: str,
    file_hashes: list[str],
    candidate_names: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve relevant chunks — strictly isolated to:
    1. The current session's uploaded files (file_hashes filter — always applied).
    2. Specific candidates if their name appears in the query (optional filter).
    """
    if not file_hashes:
        return []

    query_vector = embedding_model.encode(query).tolist()

    # Mandatory: only this session's files
    must_conditions = [
        FieldCondition(key="file_hash", match=MatchAny(any=file_hashes))
    ]

    # Optional: specific candidates
    if candidate_names:
        must_conditions.append(
            FieldCondition(
                key="candidate_name_lower",
                match=MatchAny(any=[n.lower().strip() for n in candidate_names]),
            )
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "content": p.payload["content"],
            "candidate_name": p.payload["candidate_name"],
            "section": p.payload["section"],
        }
        for p in results.points
    ]


def generate_answer(
    query: str,
    contexts: list[dict],
    available_candidates: list[str],
) -> str:
    """Generate an HR-focused answer via Groq, strictly from retrieved context."""
    if not contexts:
        return "No relevant information found in the provided CV excerpts."

    blocks = []
    for c in contexts:
        blocks.append(
            f"Candidate: {c['candidate_name']}\n"
            f"Section: {c['section']}\n"
            f"Excerpt: {c['content']}"
        )

    context_text = "\n\n---\n\n".join(blocks)
    candidates_list = ", ".join(sorted(set(available_candidates)))

    prompt = f"""You are an expert HR assistant. Answer the question using ONLY the CV excerpts provided below. Do not invent or assume any information.

Candidates in scope: {candidates_list}

Rules:
- Only use facts explicitly stated in the context below.
- Always attribute facts to the specific candidate by name.
- Use bullet points for readability.
- If the information is not in the context, clearly say so.

=== CV EXCERPTS ===
{context_text}
===================

Question: {query}

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"