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
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchText,
    PayloadSchemaType, TextIndexParams, TokenizerType,
)

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
_indexes_ensured: set[str] = set()   # collections whose payload indexes are confirmed


def get_qdrant_client() -> QdrantClient:
    """Return the Qdrant client (creates on first call)."""
    global _qdrant_client
    if _qdrant_client is None:
        url = QDRANT_URL
        api_key = QDRANT_API_KEY if QDRANT_API_KEY else None
        _qdrant_client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return _qdrant_client


def _ensure_indexes_once(collection_name: str) -> None:
    """Call _ensure_payload_indexes at most once per collection per process."""
    if collection_name not in _indexes_ensured:
        _ensure_payload_indexes(collection_name)
        _indexes_ensured.add(collection_name)


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
    _ensure_payload_indexes(collection_name)


def _ensure_payload_indexes(collection_name: str) -> None:
    """Create payload indexes for filtering and full-text search (idempotent)."""
    client = get_qdrant_client()

    # Keyword indexes for sender/recipient filtering
    for field in ("sender_email", "recipient_email"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    # Text indexes for full-text search on subject and body
    text_schema = TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=30,
        lowercase=True,
    )
    for field in ("subject", "body"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=text_schema,
            )
        except Exception:
            pass


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


# ---------------------------------------------------------------------------
# Keyword helpers (for hybrid retrieval)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    # English
    "the", "and", "for", "from", "that", "this", "with", "have", "was", "are",
    "what", "which", "when", "where", "who", "how", "mail", "email", "gmail",
    # French
    "les", "des", "une", "est", "dans", "avec", "pour", "par", "sur", "pas",
    "que", "qui", "quels", "sont", "deux", "quel", "quelle", "quelles",
    "mais", "vous", "nous", "ils", "elles", "leur", "leurs", "aux", "ces",
    "mes", "tes", "ses", "nos", "vos", "mon", "ton", "son", "une", "des",
})


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
    # Subject is repeated to give it extra weight.
    # Body includes up to 3000 chars to capture more structured content.
    att_text_for_embed = " ".join(
        (att.get("text_snippet") or "") for att in (attachments or [])
    )[:800]
    email_embed_text = (
        f"{subject or ''}\n{subject or ''}\n"
        f"{(body or '')[:3000]}\n{att_text_for_embed}"
    ).strip()
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
    Run vector search on email_embedding in Qdrant, then re-rank candidates using
    a combined score that includes body-chunk similarity (captures content beyond
    the first 1000 chars used for the email-level embedding) and attachment chunks.

        score = alpha * max(body_emb_score, max_chunk_score) + (1-alpha) * max_att_score
    """
    client = get_qdrant_client()
    q_emb = get_embedding(query)
    q_flat = q_emb.flatten()          # shape (384,)
    q_list = q_flat.tolist()

    _ensure_indexes_once(_active_collection)

    # Fetch a larger candidate pool so relevant emails aren't filtered out early
    response = client.query_points(
        collection_name=_active_collection,
        query=q_list,
        limit=k * 4,
        with_payload=True,
    )

    docs = []
    for point in response.points:
        doc = point.payload.copy()
        body_score = float(point.score)

        # Score body chunks — captures sections beyond the first 1000 chars
        # that were excluded from the email-level embedding
        max_chunk_score = 0.0
        for chunk_doc in doc.get("chunks", []):
            emb = chunk_doc.get("chunk_embedding")
            if not emb:
                continue
            chunk_emb = np.array(emb, dtype=np.float32)
            score = float(np.dot(q_flat, chunk_emb))
            if score > max_chunk_score:
                max_chunk_score = score

        # Score attachment chunks
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

        # Best content score: email embedding vs best body chunk
        best_content_score = max(body_score, max_chunk_score)
        combined = alpha * best_content_score + (1.0 - alpha) * max_att_score

        doc["score"]                    = combined
        doc["body_score"]               = body_score
        doc["chunk_score"]              = max_chunk_score
        doc["attachment_score"]         = max_att_score
        doc["best_attachment_snippet"]  = best_att_snippet
        doc["best_attachment_filename"] = best_att_filename
        docs.append(doc)

    docs.sort(key=lambda d: d["score"], reverse=True)
    return docs[:k]


# ---------------------------------------------------------------------------
# Keyword / full-text search (hybrid retrieval complement)
# ---------------------------------------------------------------------------
def _extract_keywords(text: str) -> list[str]:
    """Return meaningful tokens from *text*, excluding stopwords."""
    return [t for t in re.findall(r'\b\w{3,}\b', text.lower()) if t not in _STOPWORDS]


def keyword_search_emails(query: str, k: int = 10) -> list[dict]:
    """
    Full-text keyword search: tries Qdrant's text-index filter first (fast path).
    Falls back to Python-side scroll + substring match if the index is not ready.
    Results are re-ranked by max body-chunk similarity to the query.
    """
    tokens = _extract_keywords(query)
    if not tokens:
        return []

    qdrant = get_qdrant_client()
    q_flat = get_embedding(query).flatten()

    _ensure_indexes_once(_active_collection)

    candidates: list[dict] = []

    # Fast path: Qdrant text-index filter (requires payload index on subject/body)
    try:
        filter_conds = []
        for token in tokens[:6]:
            filter_conds.extend([
                FieldCondition(key="subject", match=MatchText(text=token)),
                FieldCondition(key="body",    match=MatchText(text=token)),
            ])
        results, _ = qdrant.scroll(
            collection_name=_active_collection,
            scroll_filter=Filter(should=filter_conds),
            limit=50,
            with_payload=True,
        )
        candidates = [point.payload for point in results]
    except Exception:
        # Index not yet available — scroll all docs and filter in Python
        offset = None
        while True:
            batch, next_offset = qdrant.scroll(
                collection_name=_active_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            if not batch:
                break
            for point in batch:
                doc = point.payload
                haystack = (doc.get("subject", "") + " " + doc.get("body", "")).lower()
                if any(t in haystack for t in tokens):
                    candidates.append(doc)
            if next_offset is None:
                break
            offset = next_offset

    # Re-rank candidates by max(body_emb_score, max_chunk_score)
    scored: list[dict] = []
    for doc in candidates:
        body_emb_list = doc.get("email_embedding", [])
        body_score = float(np.dot(q_flat, np.array(body_emb_list, dtype=np.float32))) if body_emb_list else 0.0

        max_chunk_score = 0.0
        for chunk_doc in doc.get("chunks", []):
            emb = chunk_doc.get("chunk_embedding")
            if not emb:
                continue
            score = float(np.dot(q_flat, np.array(emb, dtype=np.float32)))
            if score > max_chunk_score:
                max_chunk_score = score

        combined = max(body_score, max_chunk_score)
        out = dict(doc)
        out["score"]       = combined
        out["body_score"]  = body_score
        out["chunk_score"] = max_chunk_score
        scored.append(out)

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


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

        # Filter by contact — first try sender/recipient fields (exact match)
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

        # Fallback: if no sender/recipient match, search inside the email body.
        # This handles forwarded emails where the contact's name only appears
        # in the body thread (e.g. "Fwd:" emails forwarded by a third party).
        if not candidates:
            for doc in all_docs:
                body = doc.get("body", "").lower()
                chunks_text = " ".join(
                    c.get("chunk_text", "") for c in doc.get("chunks", [])
                ).lower()
                if contact_lower in body or contact_lower in chunks_text:
                    candidates.append(doc)

        # Re-rank by combined score (body emb + body chunks + attachments)
        q_emb_np = get_embedding(query)
        q_flat = q_emb_np.flatten()
        scored: list[tuple[float, dict]] = []

        for doc in candidates:
            body_emb = np.array(doc.get("email_embedding", []), dtype=np.float32).flatten()
            body_score = float(np.dot(q_flat, body_emb)) if body_emb.shape[0] > 0 else 0.0

            # Score body chunks to capture content beyond first 1000 chars
            max_chunk_score = 0.0
            for chunk_doc in doc.get("chunks", []):
                emb = chunk_doc.get("chunk_embedding")
                if not emb:
                    continue
                score = float(np.dot(q_flat, np.array(emb, dtype=np.float32)))
                if score > max_chunk_score:
                    max_chunk_score = score

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

            best_content_score = max(body_score, max_chunk_score)
            combined = alpha * best_content_score + (1.0 - alpha) * max_att_score
            doc["score"]                    = combined
            doc["body_score"]               = body_score
            doc["chunk_score"]              = max_chunk_score
            doc["attachment_score"]         = max_att_score
            doc["best_attachment_snippet"]  = best_att_snippet
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
