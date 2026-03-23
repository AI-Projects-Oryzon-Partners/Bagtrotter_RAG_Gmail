#!/usr/bin/env python3
"""Test script: query the RAG system with a French question."""

from dotenv import load_dotenv
load_dotenv(override=True)

import sys
from rag import ask_question

# French question from the user
question = "Quelle est la date du dernier email de Didier Beauvair concernant la connexion du site et l'intégration avec EBP ?"

print(f"[QUESTION]: {question}\n")
print("=" * 80)
print("[STATUS]: Retrieving and processing emails...\n")
sys.stdout.flush()

try:
    messages, reply = ask_question(question)
    print(f"[ANSWER]:\n{reply}\n")
    print("=" * 80)
except Exception as e:
    print(f"[ERROR]: {e}")
    import traceback
    traceback.print_exc()
