"""
email_loader.py
───────────────
Fetches emails directly from Gmail and stores them in Qdrant.

Two modes
─────────
  load_emails()   — incremental: only fetches emails newer than last run
  rebuild_index() — full wipe + re-index everything from scratch
"""

from datetime import datetime, timezone, timedelta
from email import utils as email_utils

from gmail_api import (
    authenticate_gmail,
    list_messages,
    get_message_details,
    get_last_checked_time,
    update_last_checked_time,
)
from vector_store import insert_email_record, clear_all_emails, get_qdrant_client, get_active_collection


# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------

def check_qdrant_connection() -> bool:
    """Ping Qdrant and print a clear error if the connection is broken."""
    import os
    url = os.getenv("QDRANT_URL", "http://localhost:6333")

    try:
        client = get_qdrant_client()
        # Test connection by getting collection info
        client.get_collection(get_active_collection())
        print("(LOADER): Qdrant connection OK.")
        return True
    except Exception as e:
        print(f"(LOADER): ERROR — Cannot connect to Qdrant: {e}")
        print(f"  URL used: {url!r}")
        print("  Check: Qdrant is running and accessible at the specified URL.")
        return False


# ---------------------------------------------------------------------------
# Full wipe + reset
# ---------------------------------------------------------------------------

def rebuild_index() -> None:
    """Drop the entire Qdrant emails collection and reset the last-checked timestamp."""
    if not check_qdrant_connection():
        return

    clear_all_emails()
    print(f"(REBUILD): Cleared Qdrant collection.")

    update_last_checked_time(datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    print("(REBUILD): Timestamp reset. Run load_emails() to re-index all emails.")


# ---------------------------------------------------------------------------
# Incremental Gmail -> Qdrant ingestion
# ---------------------------------------------------------------------------

def load_emails() -> None:
    """
    Fetch all inbox + sent emails newer than the last successful run,
    embed them, and store in Qdrant.

    Dedup is handled by gmail_msg_id -- re-running is always safe.
    """
    # Fail fast if Qdrant is unreachable
    if not check_qdrant_connection():
        return

    service = authenticate_gmail()

    last_checked = get_last_checked_time()
    start_date = (last_checked - timedelta(days=1)).strftime("%Y/%m/%d")
    gmail_query = f"(is:starred) after:{start_date}"

    print(f"(LOADER): Fetching emails since {last_checked.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"(LOADER): Gmail query: {gmail_query}")

    messages = list_messages(service, "me", gmail_query)

    if not messages:
        print("(LOADER): No messages returned by Gmail.")
        return

    print(f"(LOADER): {len(messages)} message(s) found -- checking for new ones...\n")

    inserted_count = 0
    skipped_count  = 0
    error_count    = 0
    latest_timestamp = last_checked

    for i, msg in enumerate(messages, 1):
        msg_id = msg["id"]
        try:
            details = get_message_details(service, "me", msg_id)
            if not details:
                print(f"  [{i}/{len(messages)}] SKIP  {msg_id} -- could not fetch details")
                skipped_count += 1
                continue

            raw_date = details.get("Date", "")
            try:
                message_datetime = email_utils.parsedate_to_datetime(raw_date)
            except Exception:
                print(f"  [{i}/{len(messages)}] SKIP  {msg_id} -- unparseable date: {raw_date!r}")
                skipped_count += 1
                continue

            if message_datetime <= last_checked:
                skipped_count += 1
                continue

            sender    = details.get("From", "")
            recipient = details.get("To", "")
            subject   = details.get("Subject", "(no subject)")
            body      = details.get("Body", "")
            attachments = details.get("Attachments", [])

            inserted = insert_email_record(
                gmail_msg_id=msg_id,
                sender=sender,
                recipient=recipient,
                date=message_datetime.isoformat(),
                subject=subject,
                body=body,
                attachments=attachments,
            )

            if not inserted:
                skipped_count += 1
                continue

            inserted_count += 1
            if message_datetime > latest_timestamp:
                latest_timestamp = message_datetime

            print(
                f"  [{i}/{len(messages)}] OK  {message_datetime.strftime('%Y-%m-%d %H:%M')}  "
                f"{subject[:60]!r}"
            )

        except Exception as e:
            error_count += 1
            print(f"  [{i}/{len(messages)}] ERROR on {msg_id}: {e}")
            continue

    print(f"\n(LOADER): Done.")
    print(f"  Inserted : {inserted_count}")
    print(f"  Skipped  : {skipped_count}  (already indexed or outside time window)")
    print(f"  Errors   : {error_count}")

    if inserted_count > 0:
        update_last_checked_time(latest_timestamp)
        print(f"  Timestamp updated to: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("  No new emails -- timestamp unchanged.")
