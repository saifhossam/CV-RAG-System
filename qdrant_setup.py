import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cv_collection")
VECTOR_SIZE = 384

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_index(field_name: str):
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"Payload index created for '{field_name}'")
    except Exception:
        print(f"Payload index for '{field_name}' already exists — skipping.")


def init_db():
    print(f"Checking for collection: {COLLECTION_NAME}...")

    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    ensure_index("file_hash")
    ensure_index("candidate_name")
    ensure_index("candidate_name_lower")
    ensure_index("section")

    print("Qdrant setup completed successfully.")


if __name__ == "__main__":
    init_db()