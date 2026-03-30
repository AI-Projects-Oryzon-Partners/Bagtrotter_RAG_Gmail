"""
Configuration and shared resources for the RAG Gmail Assistant.
Qdrant Vector Search replaces MongoDB Atlas.
Mistral LLM replaces Groq.
"""

import os
import warnings
import logging
import sys
from io import StringIO

# Suppress warnings EARLY (before any library imports that generate them)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress transformers and sentence-transformers loggers
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# Capture and suppress HF Hub print statements during import
_old_stderr = sys.stderr
sys.stderr = StringIO()

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from mistralai.client.sdk import Mistral

sys.stderr = _old_stderr



# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384
K = 6                    # Number of emails to retrieve (vector search candidates)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250     
MAX_CONTEXT_TOKENS = 16000
MAX_EMAIL_BODY_CHARS = 12000
MISTRAL_TIMEOUT_MS = int(os.getenv("MISTRAL_TIMEOUT_MS", "120000"))
# ---------------------------------------------------------------------------
# Qdrant configuration
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Collection names
EMAILS_COLLECTION = "gmails"       # one doc per email, with embedding + metadata
CHUNKS_COLLECTION = "chunks"       # one doc per chunk, with chunk-level embedding

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
load_dotenv(override=True)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")

client = Mistral(api_key=mistral_api_key, timeout_ms=MISTRAL_TIMEOUT_MS)

# ---------------------------------------------------------------------------
# Embeddings model
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
