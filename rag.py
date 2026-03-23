"""
RAG pipeline: smart query routing + LLM question-answering over emails.

Query routing logic
───────────────────
1. Temporal/contact queries  ("first time I mailed X", "last email from alice@…")
   → find_emails_by_contact()  — pure Qdrant metadata filter + date sort

2. Semantic + contact queries  ("what did we discuss about budget with Bob?")
   → hybrid_search()  — Qdrant contact filter → cosine re-rank

3. Pure semantic queries  ("what is the status of my visa application?")
   → vector_search_emails()  — Qdrant Vector Search on email_embedding
"""

import re
from datetime import datetime

from tzlocal import get_localzone

from config import client, K, MAX_CONTEXT_TOKENS, MAX_EMAIL_BODY_CHARS
from vector_store import (
    find_emails_by_contact,
    hybrid_search,
    vector_search_emails,
)


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

_TEMPORAL_FIRST_RE = re.compile(
    r"\b(first\s*time|earliest|first\s*email|first\s*mail|when\s*did\s*I\s*first|"
    r"first\s*contact|oldest\s*email|premier\s*mail|premier\s*email|premi.re\s*fois)\b",
    re.IGNORECASE,
)
_TEMPORAL_LAST_RE = re.compile(
    r"\b(last\s*time|latest|most\s*recent|last\s*email|last\s*mail|"
    r"when\s*did\s*I\s*last|most\s*recently|dernier\s*mail|dernier\s*email|derni.re\s*fois)\b",
    re.IGNORECASE,
)

_EMAIL_ADDR_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_CONTACT_PHRASE_RE = re.compile(
    r"(?:to|from|with|mail(?:ed)?|email(?:ed)?|contact(?:ed)?|wrote\s+to|"
    r"de|à|avec|envoy[eé]\s+à|re[cç]u\s+de)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
)


def extract_contact_from_query(query: str) -> str | None:
    addr_match = _EMAIL_ADDR_RE.search(query)
    if addr_match:
        return addr_match.group(0)
    name_match = _CONTACT_PHRASE_RE.search(query)
    if name_match:
        return name_match.group(1)
    return None


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

    # Use date_iso for a clean, unambiguous date in the context
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
    contact = extract_contact_from_query(query)
    is_first = bool(_TEMPORAL_FIRST_RE.search(query))
    is_last  = bool(_TEMPORAL_LAST_RE.search(query))

    # Strategy A: temporal + contact → pure metadata sort
    if contact and (is_first or is_last):
        order = "asc" if is_first else "desc"
        docs = find_emails_by_contact(contact, order=order, limit=3)
        if docs:
            return [format_email_doc(d) for d in docs]
        print(f"(RAG): No emails found for contact '{contact}', falling back to semantic search.")

    # Strategy B: contact present → hybrid (filter + cosine re-rank)
    if contact:
        docs = hybrid_search(query, contact=contact, k=k)
        if docs:
            return [format_email_doc(d) for d in docs]

    # Strategy C: pure semantic → Qdrant Vector Search
    docs = vector_search_emails(query, k=k)
    return [format_email_doc(d) for d in docs]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Tu es un assistant IA expert en analyse d'emails professionnels.
Tu as accès à une sélection d'emails pertinents fournis dans le contexte ci-dessous.
Ton rôle est de répondre aux questions de l'utilisateur en te basant UNIQUEMENT sur ces emails.

## RÈGLES STRICTES

1. **SOURCES UNIQUEMENT**
   Utilise exclusivement les emails fournis. N'invente aucune information, date, nom ou contenu absent du contexte.

2. **INFORMATION ABSENTE**
   Si la réponse ne figure pas dans les emails disponibles, dis-le explicitement :
   "Cette information ne figure pas dans les emails disponibles."
   Ne tente jamais de compléter avec des suppositions.

3. **RÉFÉRENCES PRÉCISES**
   Chaque affirmation tirée d'un email doit être accompagnée de sa source :
   - Date (utilise date_iso)
   - Expéditeur
   - Objet
   Exemple : *(Email du 2025-03-01, de alice@example.com, Objet : Réunion budget)*

4. **PLUSIEURS EMAILS PERTINENTS**
   Si plusieurs emails répondent à la question, synthétise-les en ordre chronologique et cite chaque source séparément.

5. **CITATIONS**
   Lorsqu'une citation directe apporte de la clarté, utilise-la entre guillemets et reste fidèle au texte original.

6. **CONTINUITÉ DE CONVERSATION**
   Dans une conversation multi-tours, tiens compte des échanges précédents. Si l'utilisateur précise ou reformule sa question, adapte ta réponse en conséquence.

## FORMAT DE RÉPONSE

- **Réponse directe** : commence par répondre clairement à la question.
- **Source(s)** : cite l'email ou les emails concernés (date, expéditeur, objet).
- **Citation** : extrait pertinent entre guillemets si utile.
- **Contexte** : informations complémentaires issues des emails si elles enrichissent la réponse.

## LANGUE
Réponds dans la même langue que l'utilisateur. Si l'utilisateur écrit en français, réponds en français. Si en anglais, réponds en anglais.
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

    New conversation: messages=None → retrieve emails + build system prompt.
    Continuing conversation: pass existing messages list → no new retrieval.
    """
    if messages is None:
        related_emails = smart_vector_search(question)

        local_timezone = get_localzone()
        now_str = datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

        # Build the email context block
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
            "\n\n⚠️ RAPPEL : Cite uniquement les emails listés ci-dessus. "
            "Si la réponse n'y figure pas, indique-le explicitement."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + email_context + grounding},
        ]

    messages.append({"role": "user", "content": question})

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    return messages, assistant_reply
