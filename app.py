"""
app.py - Interface Streamlit - Oryzon Partners - Analyse des correspondances clients.

Lancement :
    streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import io
import sys

import streamlit as st

# Page config
st.set_page_config(
    page_title="Oryzon Partners - Analyse Clients",
    page_icon="P",
    layout="wide",
)

# Imports fonctionnels
from email_loader import load_emails, check_qdrant_connection
from rag import ask_question
from vector_store import set_active_collection

# Collection fixe pour ce client
ACTIVE_COLLECTION = "gmails"
CLIENT_DISPLAY_NAME = "Bagtrotter"
set_active_collection(ACTIVE_COLLECTION)


if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = None
if "sync_log" not in st.session_state:
    st.session_state.sync_log = ""


with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:10px 0 4px 0;'>"
        "<span style='font-size:1.6rem;font-weight:800;letter-spacing:2px;color:#ffffff;'>ORYZON</span>"
        "<span style='font-size:1.6rem;font-weight:300;color:#ffffff;'> PARTNERS</span><br/>"
        "<span style='font-size:0.72rem;color:#6c757d;letter-spacing:1px;'>"
        "INTELLIGENCE CLIENT - ANALYSE DES CORRESPONDANCES"
        "</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Client actif")
    st.markdown(
        f"<div style='background:#f0f4ff;border-radius:8px;padding:10px 14px;margin-bottom:4px;'>"
        f"<span style='font-size:1.1rem;font-weight:700;color:#1a1a2e;'>{CLIENT_DISPLAY_NAME}</span>"
        f"<br/><span style='font-size:0.75rem;color:#6c757d;'>Collection : {ACTIVE_COLLECTION}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Synchronisation Gmail")
    if st.button("Actualiser les e-mails", use_container_width=True, type="primary",
                 help="Charge uniquement les nouveaux e-mails depuis Gmail (sync incrementale)."):
        with st.spinner("Synchronisation en cours..."):
            log_lines = []
            if not check_qdrant_connection():
                log_lines.append("ERREUR : Impossible de se connecter a Qdrant.")
            else:
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    load_emails()
                finally:
                    sys.stdout = old_stdout
                log_lines = buf.getvalue().splitlines()
                log_lines.append("Synchronisation terminee.")
            st.session_state.sync_log = "\n".join(log_lines)
        st.rerun()

    if st.session_state.sync_log:
        with st.expander("Journal de synchronisation", expanded=False):
            st.code(st.session_state.sync_log, language="")

    st.divider()

    st.subheader("Conversation")
    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.messages_display = []
        st.session_state.rag_messages = None
        st.rerun()

    st.divider()
    st.caption("2026 Oryzon Partners - Confidentiel")
    st.caption("Moteur : Mistral - Qdrant - Gmail API")


# Zone principale
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("## Analyse des Correspondances Clients")
    st.markdown(f"Explorez et interrogez les echanges e-mail avec le client **{CLIENT_DISPLAY_NAME}**.")
with col_badge:
    st.metric(label="Client actif", value=CLIENT_DISPLAY_NAME)

st.divider()

if not st.session_state.messages_display:
    st.info(
        f"Posez une question sur les echanges avec **{CLIENT_DISPLAY_NAME}**.\n\n"

    )

for msg in st.session_state.messages_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"Posez votre question sur {CLIENT_DISPLAY_NAME}..."):
    st.session_state.messages_display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            try:
                rag_messages, reply = ask_question(prompt, messages=st.session_state.rag_messages)
                st.session_state.rag_messages = rag_messages
            except Exception as e:
                print(f"(APP): Exception during ask_question: {e}")
                err_lower = str(e).lower()
                if any(k in err_lower for k in ("ssl", "handshake", "timed out", "timeout", "connection")):
                    reply = (
                        "La connexion au service IA a expiré. "
                        "Veuillez reessayer votre question dans quelques instants."
                    )
                else:
                    reply = f"Erreur : {e}"
        st.markdown(reply)

    st.session_state.messages_display.append({"role": "assistant", "content": reply})
