"""
RAG pipeline: smart query routing + LLM question-answering over emails.

Query routing logic
1. Temporal/contact queries -> `find_emails_by_contact()` using metadata sort
2. Semantic + contact queries -> `hybrid_search()` using contact-aware retrieval
3. Pure semantic queries -> vector + keyword retrieval merge
"""

import json
import re
import time
from datetime import datetime

from tzlocal import get_localzone
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from config import client, K, MAX_CONTEXT_TOKENS, MAX_EMAIL_BODY_CHARS, MISTRAL_TIMEOUT_MS
from vector_store import (
    find_emails_by_contact,
    hybrid_search,
    vector_search_emails,
    keyword_search_emails,
)


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

_EMAIL_ADDR_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
ROUTER_MODEL = "mistral-small-2603"
ROUTER_CONFIDENCE_THRESHOLD = 0.75
CHAT_RETRY_CONFIG = RetryConfig(
    strategy="backoff",
    backoff=BackoffStrategy(
        initial_interval=500,
        max_interval=3000,
        exponent=2.0,
        max_elapsed_time=10000,
    ),
    retry_connection_errors=True,
)
ROUTER_SYSTEM_PROMPT = """You classify email-search queries for routing.

Return exactly one JSON object with this schema:
{
  "intent": "temporal_contact" | "contact_semantic" | "pure_semantic",
  "time_direction": "first" | "last" | "none",
  "contact": string | null,
  "confidence": number
}

Rules:
- Choose "temporal_contact" only when the user asks for the earliest/first or latest/last email involving a specific contact.
- Choose "contact_semantic" when a specific contact is part of the query and the user asks about content/topics/discussion, not first/last chronology.
- Choose "pure_semantic" when no specific contact is required for retrieval.
- Put an email address in "contact" when explicitly present.
- If the contact is a person name, return only the minimal name string from the query.
- If there is no reliable contact, return null.
- confidence must be between 0 and 1.
- Do not include explanations, markdown, or extra keys.
"""


def debug_log(title: str, content: str | dict | list | None = None) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    if content is not None:
        if isinstance(content, (dict, list)):
            try:
                print(json.dumps(content, indent=2, ensure_ascii=False))
            except TypeError:
                print(content)
        else:
            print(content)
    print(f"{'=' * 52}")


_TRANSIENT_ERROR_KEYWORDS = ("ssl", "handshake", "timed out", "timeout", "connection")
_COMPLETE_CHAT_MAX_RETRIES = 2


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token, +4 overhead per message."""
    return sum(len(m.get("content") or "") // 4 + 4 for m in messages)


def complete_chat(step_name: str, messages: list[dict], model: str = ROUTER_MODEL):
    for attempt in range(_COMPLETE_CHAT_MAX_RETRIES + 1):
        started = time.time()
        estimated_input_tokens = _estimate_tokens(messages)
        debug_log(
            "LLM CALL",
            {
                "step": step_name,
                "model": model,
                "message_count": len(messages),
                "estimated_input_tokens": estimated_input_tokens,
                "timeout_ms": MISTRAL_TIMEOUT_MS,
                "attempt": attempt + 1,
            },
        )
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                retries=CHAT_RETRY_CONFIG,
                timeout_ms=MISTRAL_TIMEOUT_MS,
            )
            elapsed = round(time.time() - started, 2)
            usage = response.usage
            debug_log(
                "LLM CALL OK",
                {
                    "step": step_name,
                    "elapsed_sec": elapsed,
                    "tokens": {
                        "input": usage.prompt_tokens if usage else estimated_input_tokens,
                        "output": usage.completion_tokens if usage else "n/a",
                        "total": usage.total_tokens if usage else "n/a",
                    },
                },
            )
            return response
        except Exception as exc:
            elapsed = round(time.time() - started, 2)
            exc_lower = str(exc).lower()
            is_transient = any(k in exc_lower for k in _TRANSIENT_ERROR_KEYWORDS)
            if is_transient and attempt < _COMPLETE_CHAT_MAX_RETRIES:
                wait_sec = 2 ** attempt  # 1s, then 2s
                debug_log(
                    "LLM RETRY",
                    {
                        "step": step_name,
                        "attempt": attempt + 1,
                        "wait_sec": wait_sec,
                        "error": str(exc),
                    },
                )
                time.sleep(wait_sec)
                continue
            debug_log(
                "LLM CALL FAILED",
                {
                    "step": step_name,
                    "elapsed_sec": elapsed,
                    "error": str(exc),
                },
            )
            raise


def extract_contact_from_query(query: str) -> str | None:
    addr_match = _EMAIL_ADDR_RE.search(query)
    if addr_match:
        return addr_match.group(0)
    return None


def _extract_json_object(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw_text[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def classify_query_intent(query: str) -> dict:
    fallback_contact = extract_contact_from_query(query)
    fallback = {
        "intent": "pure_semantic",
        "time_direction": "none",
        "contact": fallback_contact,
        "confidence": 0.0,
    }

    try:
        response = complete_chat(
            step_name="router_classification",
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw_content = response.choices[0].message.content or ""
        parsed = _extract_json_object(raw_content)
        if not parsed:
            print("(RAG router): invalid classifier JSON, using fallback semantic routing.")
            debug_log("ROUTER FALLBACK", fallback)
            return fallback

        intent = parsed.get("intent", "pure_semantic")
        if intent not in {"temporal_contact", "contact_semantic", "pure_semantic"}:
            intent = "pure_semantic"

        time_direction = parsed.get("time_direction", "none")
        if time_direction not in {"first", "last", "none"}:
            time_direction = "none"

        contact = parsed.get("contact")
        if contact is not None:
            contact = str(contact).strip() or None

        if fallback_contact and not contact:
            contact = fallback_contact

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        if intent == "temporal_contact":
            if not contact or time_direction == "none":
                return fallback
        elif intent == "contact_semantic":
            if not contact:
                return fallback
        else:
            time_direction = "none"

        return {
            "intent": intent,
            "time_direction": time_direction,
            "contact": contact,
            "confidence": confidence,
        }
    except Exception as exc:
        print(f"(RAG router): classifier failed, using fallback semantic routing. Error: {exc}")
        debug_log("ROUTER FALLBACK", fallback)
        return fallback


# ---------------------------------------------------------------------------
# Email formatter
# ---------------------------------------------------------------------------
def format_email_doc(doc: dict, max_body: int = MAX_EMAIL_BODY_CHARS) -> str:
    body = doc.get("body", "")
    if len(body) > max_body:
        chunks = doc.get("chunks", [])
        if chunks:
            body = "\n...\n".join(c["chunk_text"] for c in chunks)
        if len(body) > max_body:
            body = body[:max_body] + "\n... [truncated]"

    date_display = doc.get("date_iso") or doc.get("date", "")

    return (
        f"<Email Start>\n"
        f"Date and Time: {date_display}\n"
        f"Sender: {doc.get('sender', '')}\n"
        f"To: {doc.get('recipient', '')}\n"
        f"Subject: {doc.get('subject', '')}\n"
        f"Body:\n{body}\n"
        f"<Email End>\n"
    )


# ---------------------------------------------------------------------------
# Smart vector search
# ---------------------------------------------------------------------------
def smart_vector_search(query: str, k: int = K) -> list[str]:
    route = classify_query_intent(query)
    contact = route["contact"]
    route_confident = route["confidence"] >= ROUTER_CONFIDENCE_THRESHOLD
    explicit_email_contact = bool(contact and "@" in contact)

    debug_log("ROUTER DECISION", route)

    if (
        route["intent"] == "temporal_contact"
        and contact
        and route["time_direction"] in {"first", "last"}
        and route_confident
    ):
        order = "asc" if route["time_direction"] == "first" else "desc"
        debug_log(
            "RETRIEVAL STRATEGY",
            {
                "strategy": "find_emails_by_contact",
                "contact": contact,
                "order": order,
                "limit": 3,
            },
        )
        docs = find_emails_by_contact(contact, order=order, limit=3)
        if docs:
            debug_log(
                "TEMPORAL RESULTS",
                [
                    {
                        "date_iso": d.get("date_iso"),
                        "sender": d.get("sender"),
                        "recipient": d.get("recipient"),
                        "subject": d.get("subject"),
                    }
                    for d in docs
                ],
            )
            return [format_email_doc(d) for d in docs]
        print(f"(RAG): No emails found for contact '{contact}', falling back to semantic search.")

    if contact and (route_confident or explicit_email_contact):
        debug_log(
            "RETRIEVAL STRATEGY",
            {
                "strategy": "hybrid_search",
                "contact": contact,
                "k": k,
                "reason": "confident_contact_route" if route_confident else "explicit_email_contact",
            },
        )
        docs = hybrid_search(query, contact=contact, k=k)
        if docs:
            debug_log(
                "HYBRID FILTERED RESULTS",
                [
                    {
                        "score": d.get("score"),
                        "date_iso": d.get("date_iso"),
                        "sender": d.get("sender"),
                        "recipient": d.get("recipient"),
                        "subject": d.get("subject"),
                    }
                    for d in docs[:8]
                ],
            )
            return [format_email_doc(d) for d in docs]

    debug_log(
        "RETRIEVAL STRATEGY",
        {
            "strategy": "vector_search + keyword_search merge",
            "vector_k": k,
            "keyword_k": max(k // 2, 5),
        },
    )
    vector_docs = vector_search_emails(query, k=k)
    keyword_docs = keyword_search_emails(query, k=max(k // 2, 5))

    seen: dict[str, dict] = {}
    for doc in vector_docs:
        mid = doc.get("gmail_msg_id", "")
        if mid:
            seen[mid] = doc

    for doc in keyword_docs:
        mid = doc.get("gmail_msg_id", "")
        if not mid:
            continue
        if mid not in seen:
            seen[mid] = doc
        else:
            seen[mid]["score"] = seen[mid]["score"] + 0.05

    merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)

    print(
        f"\n(RAG hybrid): {len(vector_docs)} vector + {len(keyword_docs)} keyword "
        f"-> {len(merged)} merged candidates"
    )
    for i, d in enumerate(merged[:8]):
        print(
            f"  [{i + 1}] score={d.get('score', 0):.3f} "
            f"chunk={d.get('chunk_score', 0):.3f} "
            f"body={d.get('body_score', 0):.3f} "
            f"subj={d.get('subject', '')[:70]}"
        )

    debug_log(
        "FINAL RETRIEVAL RESULTS",
        [
            {
                "score": d.get("score"),
                "chunk_score": d.get("chunk_score"),
                "body_score": d.get("body_score"),
                "date_iso": d.get("date_iso"),
                "sender": d.get("sender"),
                "recipient": d.get("recipient"),
                "subject": d.get("subject"),
            }
            for d in merged[:k]
        ],
    )

    return [format_email_doc(d) for d in merged[:k]]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Tu es un assistant IA expert en analyse d'emails professionnels.
Tu as acces a une selection d'emails pertinents fournis dans le contexte ci-dessous.
Ton role est de repondre aux questions de l'utilisateur en te basant UNIQUEMENT sur ces emails.

## REGLES STRICTES

1. **SOURCES UNIQUEMENT**
   Utilise exclusivement les emails fournis. N'invente aucune information, date, nom ou contenu absent du contexte.

2. **INFORMATION ABSENTE**
   Si la reponse ne figure pas dans les emails disponibles, dis-le explicitement :
   "Cette information ne figure pas dans les emails disponibles."
   Ne tente jamais de completer avec des suppositions.

3. **REFERENCES PRECISES**
   Chaque affirmation tiree d'un email doit etre accompagnee de sa source :
   - Date (utilise date_iso)
   - Expediteur
   - Objet
   Exemple : *(Email du 2025-03-01, de alice@example.com, Objet : Reunion budget)*

4. **PLUSIEURS EMAILS PERTINENTS**
   Si plusieurs emails repondent a la question, synthetise-les en ordre chronologique et cite chaque source separement.

5. **CITATIONS**
   Lorsqu'une citation directe apporte de la clarte, utilise-la entre guillemets et reste fidele au texte original.

6. **CONTINUITE DE CONVERSATION**
   Dans une conversation multi-tours, tiens compte des echanges precedents. Si l'utilisateur precise ou reformule sa question, adapte ta reponse en consequence.

## FORMAT DE REPONSE

- **Reponse directe** : commence par repondre clairement a la question.
- **Source(s)** : cite l'email ou les emails concernes (date, expediteur, objet).
- **Citation** : extrait pertinent entre guillemets si utile.
- **Contexte** : informations complementaires issues des emails si elles enrichissent la reponse.

## LANGUE
Reponds dans la meme langue que l'utilisateur. Si l'utilisateur ecrit en francais, reponds en francais. Si en anglais, reponds en anglais.
\
"""


# ---------------------------------------------------------------------------
# Ask question (RAG)
# ---------------------------------------------------------------------------
def ask_question(
    question: str,
    messages: list[dict] | None = None,
) -> tuple[list[dict], str]:
    """
    Send a user question to the LLM, enriched with relevant email context.

    Retrieval runs on EVERY question so each turn gets the most relevant emails.
    Multi-turn conversation history (previous Q&A pairs) is preserved across turns.
    """
    # Always retrieve fresh context for this specific question
    related_emails = smart_vector_search(question)

    local_timezone = get_localzone()
    now_str = datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

    email_lines = [f"Date et heure actuelles : {now_str}\n"]
    included = 0
    token_estimate = len(SYSTEM_PROMPT) // 3 + 50
    for email in related_emails:
        email_tokens = len(email) // 3
        if token_estimate + email_tokens > MAX_CONTEXT_TOKENS:
            break
        email_lines.append(f"Email({included + 1}):\n{email}")
        token_estimate += email_tokens
        included += 1

    email_context = (
        f"EMAILS DISPONIBLES ({included} au total) :\n\n"
        + "\n".join(email_lines)
    )

    grounding = (
        "\n\nRAPPEL : Cite uniquement les emails listes ci-dessus. "
        "Si la reponse n'y figure pas, indique-le explicitement."
    )

    system_content = SYSTEM_PROMPT + "\n\n" + email_context + grounding

    if messages is None:
        # New conversation — start fresh
        messages = [{"role": "system", "content": system_content}]
    else:
        # Continuing conversation — update system message with fresh context,
        # keep existing Q&A history (assistant/user turns) intact
        messages[0] = {"role": "system", "content": system_content}

    messages.append({"role": "user", "content": question})

    response = complete_chat(
        step_name="answer_generation",
        model=ROUTER_MODEL,
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    return messages, assistant_reply
