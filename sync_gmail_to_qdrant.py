"""
sync_gmail_to_qdrant.py
───────────────────────
Standalone pipeline: fetch emails from Gmail → embed → push to Qdrant.

Usage
─────
  python sync_gmail_to_qdrant.py            # incremental sync (new emails only)
  python sync_gmail_to_qdrant.py --rebuild  # wipe collection & re-index everything
  python sync_gmail_to_qdrant.py --stats    # show collection stats then exit

Flags
─────
  --rebuild   Drop the Qdrant collection, reset the timestamp, re-index all mail.
  --stats     Print collection document/vector counts and exit.
  --help      Show this help message.
"""

import sys

from dotenv import load_dotenv

load_dotenv(override=True)

# ── late imports so that .env is loaded before config.py reads env vars ──────
from email_loader import load_emails, rebuild_index, check_qdrant_connection
from vector_store import get_qdrant_client
from config import EMAILS_COLLECTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_stats() -> None:
    """Print basic Qdrant collection stats."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(EMAILS_COLLECTION)
        print(f"\n── Collection: {EMAILS_COLLECTION} ──")
        print(f"  Points (emails) : {info.points_count}")
        print(f"  Indexed vectors : {info.indexed_vectors_count}")
        print(f"  Status          : {info.status}")
    except Exception as e:
        print(f"(STATS): Could not retrieve stats — {e}")


def print_help() -> None:
    print(__doc__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_help()
        return

    if "--stats" in args:
        if not check_qdrant_connection():
            sys.exit(1)
        show_stats()
        return

    if not check_qdrant_connection():
        sys.exit(1)

    if "--rebuild" in args:
        print("(SYNC): Rebuild mode — wiping collection and re-indexing all emails.\n")
        rebuild_index()

    print("(SYNC): Starting Gmail → Qdrant sync...\n")
    load_emails()
    print("\n(SYNC): Sync complete.")

    # Always show a short summary after a successful sync
    show_stats()


if __name__ == "__main__":
    main()
