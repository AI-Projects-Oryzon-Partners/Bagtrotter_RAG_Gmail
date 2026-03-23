"""
Qdrant Vector Search store.

Schema (one document per email in `emails` collection):
{
    "id":                  str (UUID),
    "gmail_msg_id":        str,          # dedup key
    "sender":              str,          # full "Name <addr>" string
    "sender_email":        str,          # extracted addr only — indexed for filtering
    "recipient":           str,
    "recipient_email":     str,
    "date":                str,          # original RFC 2822 string
    "date_iso":            str,          # ISO 8601 — used for sort/range queries
    "subject":             str,
    "body":                str,
    "email_embedding":     list[float],  # 384-dim, L2-normalised
    "chunks": [                          # email body sub-documents
        {
            "chunk_index": int,
            "chunk_text":  str,
            "chunk_embedding": list[float],
        }
    ],
    "attachments": [                     # attachment metadata (no full text stored twice)
        {
            "filename":     str,
            "mimeType":     str,
            "size":         int,
            "text_snippet": str,
        }
    ],
    "attachment_chunks": [               # embedded attachment sub-documents
        {
            "source":          "attachment",
            "filename":        str,
            "mimeType":        str,
            "chunk_index":     int,
            "chunk_text":      str,
            "chunk_embedding": list[float],
        }
    ]
}
"""

import re
import uuid
from datetime import datetime, timezone
from email import utils as email_utils
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    QDRANT_URL, QDRANT_API_KEY,
    EMAILS_COLLECTION,
    EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
    embeddings,
)


# ---------------------------------------------------------------------------
# Active collection (dynamic — changed at runtime by the Streamlit UI)
# ---------------------------------------------------------------------------
_active_collection: str = EMAILS_COLLECTION


def set_active_collection(name: str) -> None:
    """Switch the active Qdrant collection for all subsequent operations."""
    global _active_collection
    _active_collection = name


def get_active_collection() -> str:
    """Return the currently active Qdrant collection name."""
    return _active_collection


# ---------------------------------------------------------------------------
# Qdrant connection (lazy singleton)
# ---------------------------------------------------------------------------
_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the Qdrant client (creates on first call)."""
    global _qdrant_client
    if _qdrant_client is None:
        url = QDRANT_URL
        api_key = QDRANT_API_KEY if QDRANT_API_KEY else None
        _qdrant_client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return _qdrant_client


def _ensure_collection(collection_name: str) -> None:
    """Create *collection_name* in Qdrant if it doesn't already exist."""
    client = get_qdrant_client()
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """Return a normalised (1, D) float32 array for cosine similarity."""
    emb = np.array(embeddings.embed_query(text), dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def embedding_to_list(emb: np.ndarray) -> list[float]:
    """Convert (1, D) numpy array → plain Python list for Qdrant storage."""
    return emb.flatten().tolist()


# ---------------------------------------------------------------------------
# Address extraction helper
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")


def extract_email_address(full_addr: str) -> str:
    """Return the bare email address from strings like 'Alice <alice@example.com>'."""
    if not full_addr:
        return ""
    match = _EMAIL_RE.search(full_addr)
    return match.group(0).lower() if match else full_addr.lower().strip()


def parse_date_iso(date_str: str) -> str:
    """Parse RFC 2822 or ISO 8601 date string to normalised ISO 8601 UTC string.
    Returns empty string on failure."""
    if not date_str:
        return ""
    # Try RFC 2822 first (e.g. "Mon, 01 Jan 2024 10:00:00 +0000")
    try:
        dt = email_utils.parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    # Try ISO 8601 / datetime string (e.g. "2026-02-26 16:00:16+00:00")
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks. Short texts become a single chunk."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start + overlap >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------
def insert_email_record(
    gmail_msg_id: str,
    sender: str,
    recipient: str,
    date: str,
    subject: str,
    body: str,
    attachments: list[dict] | None = None,
) -> bool:
    """
    Build chunk embeddings + email-level embedding and upsert a single document
    into the Qdrant `emails` collection.

    Returns False if the email was already indexed (duplicate gmail_msg_id).
    """
    _ensure_collection(_active_collection)
    client = get_qdrant_client()

    # Dedup check
    try:
        results = client.scroll(
            collection_name=_active_collection,
            limit=1,
            scroll_filter={
                "must": [
                    {
                        "key": "gmail_msg_id",
                        "match": {"value": gmail_msg_id},
                    }
                ]
            },
        )
        if results[0]:  # If any results found
            return False
    except Exception:
        pass

    # ---- Email-level embedding (subject + body + attachment text) ----
    # Including attachment snippets means Qdrant's native search finds PDF content
    # without any expensive scroll at query time.
    att_text_for_embed = " ".join(
        (att.get("text_snippet") or "") for att in (attachments or [])
    )[:800]
    email_embed_text = f"{subject or ''}\n{(body or '')[:1000]}\n{att_text_for_embed}".strip()
    email_emb = get_embedding(email_embed_text)

    # ---- Chunk-level embeddings ----
    raw_chunks = chunk_text(body or "")
    chunk_docs: list[dict[str, Any]] = []
    for i, chunk in enumerate(raw_chunks):
        chunk_embed_text = f"{subject or ''}\n{chunk}"
        chunk_emb = get_embedding(chunk_embed_text)
        chunk_docs.append({
            "chunk_index":     i,
            "chunk_text":      chunk,
            "chunk_embedding": embedding_to_list(chunk_emb),
        })

    # ---- Attachment-level chunks (PDF / Excel / CSV / Google Sheet) ----
    attachments = attachments or []
    attachment_meta: list[dict[str, Any]] = []
    attachment_chunk_docs: list[dict[str, Any]] = []

    for att in attachments:
        att_text = att.get("text_full", "") or att.get("text_snippet", "")
        att_filename = att.get("filename", "")
        att_mime = att.get("mimeType", "")
        att_size = att.get("size", 0)

        # Store metadata (no full text duplicated at root level)
        attachment_meta.append({
            "filename":     att_filename,
            "mimeType":     att_mime,
            "size":         att_size,
            "text_snippet": att.get("text_snippet", ""),
        })

        # Embed attachment text as separate chunks
        if att_text.strip():
            for i, chunk in enumerate(chunk_text(att_text)):
                chunk_embed_text = f"{subject or ''}\n{chunk}"
                chunk_emb = get_embedding(chunk_embed_text)
                attachment_chunk_docs.append({
                    "source":          "attachment",
                    "filename":        att_filename,
                    "mimeType":        att_mime,
                    "chunk_index":     i,
                    "chunk_text":      chunk,
                    "chunk_embedding": embedding_to_list(chunk_emb),
                })

    doc = {
        "gmail_msg_id":     gmail_msg_id,
        "sender":           sender or "",
        "sender_email":     extract_email_address(sender or ""),
        "recipient":        recipient or "",
        "recipient_email":  extract_email_address(recipient or ""),
        "date":             date or "",
        "date_iso":         parse_date_iso(date or ""),
        "subject":          subject or "",
        "body":             body or "",
        "email_embedding":  embedding_to_list(email_emb),
        "chunks":           chunk_docs,
        "attachments":      attachment_meta,
        "attachment_chunks": attachment_chunk_docs,
    }

    # Generate unique ID for this email
    point_id = str(uuid.uuid4())

    # Upsert into Qdrant
    point = PointStruct(
        id=point_id,
        vector=embedding_to_list(email_emb),
        payload=doc,
    )
    client.upsert(
        collection_name=_active_collection,
        points=[point],
    )
    return True


# ---------------------------------------------------------------------------
# Vector search (email-level — used for general semantic queries)
# ---------------------------------------------------------------------------
def vector_search_emails(query: str, k: int = 20, alpha: float = 0.6) -> list[dict]:
    """
    Run vector search on email_embedding in Qdrant (which includes attachment text),
    then re-rank the returned candidates using a combined score:

        score_combined = alpha * score_body + (1 - alpha) * max(attachment_chunk_scores)

    Attachment chunks are only scored for the top candidates returned by Qdrant,
    so this is fast regardless of collection size.
    """
    client = get_qdrant_client()
    q_emb = get_embedding(query)
    q_flat = q_emb.flatten()          # shape (384,)
    q_list = q_flat.tolist()

    # Single fast Qdrant query — email embedding already encodes attachment content
    response = client.query_points(
        collection_name=_active_collection,
        query=q_list,
        limit=k * 2,
        with_payload=True,
    )

    docs = []
    for point in response.points:
        doc = point.payload.copy()
        body_score = float(point.score)

        # Score attachment chunks only for these top candidates (fast)
        max_att_score = 0.0
        best_att_snippet = ""
        best_att_filename = ""

        for att_chunk in doc.get("attachment_chunks", []):
            emb = att_chunk.get("chunk_embedding")
            if not emb:
                continue
            att_emb = np.array(emb, dtype=np.float32)
            score = float(np.dot(q_flat, att_emb))
            if score > max_att_score:
                max_att_score = score
                best_att_snippet = att_chunk.get("chunk_text", "")[:400]
                best_att_filename = att_chunk.get("filename", "")

        combined = alpha * body_score + (1.0 - alpha) * max_att_score

        doc["score"]                    = combined
        doc["body_score"]               = body_score
        doc["attachment_score"]         = max_att_score
        doc["best_attachment_snippet"]  = best_att_snippet
        doc["best_attachment_filename"] = best_att_filename
        docs.append(doc)

    docs.sort(key=lambda d: d["score"], reverse=True)
    return docs[:k]


# ---------------------------------------------------------------------------
# Filtered temporal search (attribute + date queries)
# ---------------------------------------------------------------------------
def find_emails_by_contact(
    contact: str,
    order: str = "asc",     # "asc" → earliest first, "desc" → latest first
    limit: int = 5,
) -> list[dict]:
    """
    Return emails involving *contact* (sender or recipient), sorted by date.
    Used for "first/last time I mailed X" queries — pure metadata, no vector search.
    """
    client = get_qdrant_client()
    contact_lower = contact.lower().strip()

    # Scroll through all documents and filter in Python
    all_docs = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=_active_collection,
            limit=100,
            offset=offset,
        )
        if not results:
            break
        all_docs.extend([point.payload for point in results])
        if next_offset is None:
            break
        offset = next_offset

    # Filter by contact
    filtered = []
    for doc in all_docs:
        sender_email = doc.get("sender_email", "").lower()
        recipient_email = doc.get("recipient_email", "").lower()
        sender = doc.get("sender", "").lower()
        recipient = doc.get("recipient", "").lower()

        if (contact_lower in sender_email or
            contact_lower in recipient_email or
            contact_lower in sender or
            contact_lower in recipient):
            filtered.append(doc)

    # Sort by date
    def _sort_key(doc):
        return doc.get("date_iso") or doc.get("date") or ""

    filtered.sort(key=_sort_key, reverse=(order == "desc"))
    return filtered[:limit]


# ---------------------------------------------------------------------------
# Hybrid search (semantic + optional contact filter)
# ---------------------------------------------------------------------------
def hybrid_search(
    query: str,
    contact: str | None = None,
    k: int = 20,
    alpha: float = 0.6,
) -> list[dict]:
    """
    If *contact* is given: filter by sender/recipient, then re-rank
    the candidates by combined score (body + attachments).
    Otherwise: use vector_search_emails which includes attachment scoring.
    """
    if contact:
        # Get all documents and filter by contact
        client = get_qdrant_client()
        contact_lower = contact.lower().strip()

        all_docs = []
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=_active_collection,
                limit=100,
                offset=offset,
            )
            if not results:
                break
            all_docs.extend([point.payload for point in results])
            if next_offset is None:
                break
            offset = next_offset

        # Filter by contact
        candidates = []
        for doc in all_docs:
            sender_email = doc.get("sender_email", "").lower()
            recipient_email = doc.get("recipient_email", "").lower()
            sender = doc.get("sender", "").lower()
            recipient = doc.get("recipient", "").lower()

            if (contact_lower in sender_email or
                contact_lower in recipient_email or
                contact_lower in sender or
                contact_lower in recipient):
                candidates.append(doc)

        # Re-rank by combined score (body + attachments)
        q_emb_np = get_embedding(query)
        q_flat = q_emb_np.flatten()
        scored: list[tuple[float, dict]] = []

        for doc in candidates:
            body_emb = np.array(doc.get("email_embedding", []), dtype=np.float32).flatten()
            body_score = float(np.dot(q_flat, body_emb)) if body_emb.shape[0] > 0 else 0.0

            max_att_score = 0.0
            best_att_snippet = ""
            best_att_filename = ""
            for att_chunk in doc.get("attachment_chunks", []):
                emb = att_chunk.get("chunk_embedding")
                if not emb:
                    continue
                att_emb = np.array(emb, dtype=np.float32)
                score = float(np.dot(q_flat, att_emb))
                if score > max_att_score:
                    max_att_score = score
                    best_att_snippet = att_chunk.get("chunk_text", "")[:400]
                    best_att_filename = att_chunk.get("filename", "")

            combined = alpha * body_score + (1.0 - alpha) * max_att_score
            doc["score"] = combined
            doc["body_score"] = body_score
            doc["attachment_score"] = max_att_score
            doc["best_attachment_snippet"] = best_att_snippet
            doc["best_attachment_filename"] = best_att_filename
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    # No contact filter — use vector search with attachment scoring
    return vector_search_emails(query, k=k, alpha=alpha)


# ---------------------------------------------------------------------------
# Utility: Clear all emails (for rebuild)
# ---------------------------------------------------------------------------
def clear_all_emails() -> int:
    """Delete all documents from the emails collection. Returns count deleted."""
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=_active_collection)
        # Recreate the collection
        client.create_collection(
            collection_name=_active_collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        return 0  # Collection was recreated, count is implicit
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return 0
