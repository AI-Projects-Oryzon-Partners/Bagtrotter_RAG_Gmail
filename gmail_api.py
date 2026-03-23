"""
Gmail API: authentication, message listing, and message detail extraction.
Supports PDF, Excel, CSV attachment extraction and public Google Sheets links.
"""

import base64
import io
import os
import re
from datetime import datetime, timezone
from typing import Any

import dateutil.parser
import requests as http_requests
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Attachment extraction constants
# ---------------------------------------------------------------------------
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB per attachment
MAX_SPREADSHEET_ROWS = 500              # rows to read from Excel/CSV

# MIME types we know how to parse
_SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.ms-excel",                                            # xls
    "text/csv",
    "application/csv",
}

# ---------------------------------------------------------------------------
# Gmail scopes
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def authenticate_gmail():
    """Authenticate with Google and return a Gmail API service object."""
    creds = None
    token_file = "token.json"

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except ValueError:
            print("Token file is corrupted, deleting and reauthenticating...")
            os.remove(token_file)
            creds = None

    if creds and creds.expired and creds.refresh_token:
        print("Refreshing expired token...")
        try:
            creds.refresh(Request())
            with open(token_file, "w") as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"Token refresh failed ({e}), re-authenticating...")
            os.remove(token_file)
            creds = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        ports_to_try = [8080, 5000, 3000, 8888, 9000]
        creds = None
        for port in ports_to_try:
            try:
                # access_type='offline'  → forces Google to return a refresh_token
                # prompt='consent'       → ensures refresh_token even on re-auth
                creds = flow.run_local_server(
                    port=port, access_type="offline", prompt="consent"
                )
                print(f"Successfully authenticated using port {port}")
                break
            except (PermissionError, OSError):
                print(f"Port {port} is not available, trying next port...")
                continue

        if creds is None:
            print("Using automatic port selection...")
            creds = flow.run_local_server(
                port=0, access_type="offline", prompt="consent"
            )

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


# ---------------------------------------------------------------------------
# Attachment text extractors
# ---------------------------------------------------------------------------

def _extract_pdf_text(data: bytes) -> str:
    """Return text extracted from PDF bytes using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        return f"(PDF extraction failed: {e})"


def _extract_spreadsheet_text(data: bytes, mime: str) -> str:
    """Return CSV-like text from Excel or CSV bytes."""
    try:
        import pandas as pd
        bio = io.BytesIO(data)
        if "spreadsheetml" in mime or "excel" in mime:
            dfs = pd.read_excel(bio, sheet_name=None)
            parts = []
            for sheet_name, df in dfs.items():
                parts.append(f"[Sheet: {sheet_name}]")
                parts.append(df.head(MAX_SPREADSHEET_ROWS).to_csv(index=False))
            return "\n".join(parts)
        else:  # csv
            df = pd.read_csv(bio)
            return df.head(MAX_SPREADSHEET_ROWS).to_csv(index=False)
    except Exception as e:
        return f"(Spreadsheet extraction failed: {e})"


def _detect_google_sheet_links(text: str) -> list[str]:
    """Find Google Sheets URLs inside email body text."""
    return re.findall(
        r"https://docs\.google\.com/spreadsheets/d/[a-zA-Z0-9\-_]+",
        text or "",
    )


def _fetch_public_sheet_text(sheet_url: str) -> str | None:
    """Download a *public* Google Sheet as CSV text. Returns None if not public."""
    try:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9\-_]+)", sheet_url)
        if not m:
            return None
        export_url = (
            f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
        )
        resp = http_requests.get(export_url, timeout=10)
        if resp.status_code == 200:
            return resp.text[:MAX_SPREADSHEET_ROWS * 200]  # safety cap
    except Exception:
        pass
    return None


def _get_attachment_list(
    service, user_id: str, msg_id: str, parts: list
) -> list[dict[str, Any]]:
    """
    Recursively walk MIME parts, download supported attachments, extract text.
    Returns a list of attachment dicts:
        {filename, mimeType, size, text_snippet, text_full}
    """
    attachments: list[dict[str, Any]] = []

    for part in parts:
        mime = part.get("mimeType", "")
        filename = part.get("filename", "")
        body = part.get("body", {})

        # Recurse into nested multipart
        if "parts" in part:
            attachments.extend(
                _get_attachment_list(service, user_id, msg_id, part["parts"])
            )
            continue

        att_id = body.get("attachmentId")
        size = body.get("size", 0)

        if not att_id or not filename:
            continue
        if mime not in _SUPPORTED_MIMES:
            continue
        if size > MAX_ATTACHMENT_SIZE:
            print(f"  (ATT): Skipping {filename!r} — too large ({size} bytes)")
            continue

        try:
            att_data = (
                service.users()
                .messages()
                .attachments()
                .get(userId=user_id, messageId=msg_id, id=att_id)
                .execute()
            )
            raw = base64.urlsafe_b64decode(att_data.get("data", ""))
        except Exception as e:
            print(f"  (ATT): Failed to download {filename!r}: {e}")
            continue

        if mime == "application/pdf":
            text = _extract_pdf_text(raw)
        elif "spreadsheetml" in mime or "excel" in mime or "csv" in mime:
            text = _extract_spreadsheet_text(raw, mime)
        else:
            text = ""

        attachments.append({
            "filename": filename,
            "mimeType": mime,
            "size": size,
            "text_snippet": text[:500],
            "text_full": text,
        })

    return attachments


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------
def clean_html(html_content: str) -> str:
    """Clean HTML content and extract plain text."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text("\n", strip=True)


# ---------------------------------------------------------------------------
# MIME body extraction
# ---------------------------------------------------------------------------
def get_plain_text_body(parts: list) -> str | None:
    """Recursively extract plain text from MIME parts, falling back to HTML."""
    plain_text = None
    html_text = None

    for part in parts:
        mime_type = part["mimeType"]
        if "parts" in part:
            text = get_plain_text_body(part["parts"])
            if text:
                return text
        elif mime_type == "text/plain" and "data" in part["body"]:
            plain_text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
        elif mime_type == "text/html" and "data" in part["body"]:
            html_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
            html_text = clean_html(html_body)

    return plain_text if plain_text else html_text


# ---------------------------------------------------------------------------
# Message details
# ---------------------------------------------------------------------------
def get_message_details(service, user_id: str, msg_id: str) -> dict | None:
    """Fetch full message details (From, Cc, Subject, Date, Body)."""
    try:
        message = (
            service.users()
            .messages()
            .get(userId=user_id, id=msg_id, format="full")
            .execute()
        )
        headers = message["payload"]["headers"]
        details = {
            header["name"]: header["value"]
            for header in headers
            if header["name"] in ["From", "To", "Cc", "Subject", "Date"]
        }

        payload = message["payload"]
        if "parts" in payload:
            details["Body"] = get_plain_text_body(payload["parts"])
            details["Attachments"] = _get_attachment_list(
                service, user_id, msg_id, payload["parts"]
            )
        elif "data" in payload["body"]:
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
            details["Body"] = clean_html(body)
            details["Attachments"] = []
        else:
            details["Body"] = None
            details["Attachments"] = []

        # ── Detect public Google Sheets links in body and fetch them ──────────
        sheet_links = _detect_google_sheet_links(details.get("Body", "") or "")
        for link in sheet_links[:3]:  # cap at 3 links per email
            text = _fetch_public_sheet_text(link)
            if text:
                details["Attachments"].append({
                    "filename": link,
                    "mimeType": "text/csv",
                    "size": len(text),
                    "text_snippet": text[:500],
                    "text_full": text,
                })

        if details["Attachments"]:
            print(
                f"  (ATT): {len(details['Attachments'])} attachment(s) extracted for "
                f"{msg_id}"
            )

        return details
    except Exception as error:
        print(f"An error occurred: {error}")
        return None


# ---------------------------------------------------------------------------
# Message listing
# ---------------------------------------------------------------------------
def list_messages(service, user_id: str, query: str = "", max_results: int = 500) -> list | None:
    """List messages matching the given query (up to *max_results*)."""
    try:
        messages = []
        request = service.users().messages().list(userId=user_id, q=query)
        while request is not None:
            response = request.execute()
            if "messages" in response:
                messages.extend(response["messages"])
            if len(messages) >= max_results:
                messages = messages[:max_results]
                break
            request = service.users().messages().list_next(request, response)
        return messages
    except Exception as error:
        print(f"An error occurred: {error}")
        return None


# ---------------------------------------------------------------------------
# Timestamp tracking helpers
# ---------------------------------------------------------------------------
def get_last_checked_time() -> datetime:
    """Return the datetime of the last fetched email."""
    try:
        with open("last_checked.txt", "r") as file:
            return dateutil.parser.parse(file.read().strip())
    except FileNotFoundError:
        return datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def update_last_checked_time(timestamp: datetime) -> None:
    """Persist the timestamp of the most recent fetched email."""
    with open("last_checked.txt", "w") as file:
        file.write(str(timestamp))
