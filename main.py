"""
RAG Gmail Assistant - entry point.

Usage:
    python main.py              # load emails then start text chat
    python main.py --load-only  # only load/index new emails
    python main.py --rebuild    # delete old index & DB, re-index all emails
"""

# Load .env FIRST before any other imports read os.getenv()
from dotenv import load_dotenv
load_dotenv(override=True)

import sys

from email_loader import load_emails, rebuild_index
from chat import start_chat


def main() -> None:
    if "--rebuild" in sys.argv:
        rebuild_index()

    load_emails()

    if "--load-only" in sys.argv or "--rebuild" in sys.argv:
        print("(System): Emails loaded. Exiting.")
        return

    start_chat()


if __name__ == "__main__":
    main()