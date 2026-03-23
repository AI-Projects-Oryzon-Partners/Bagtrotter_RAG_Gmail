# RAG-Gmail-Assistant

## Description
RAG-Gmail-Assistant is a Retrieval-Augmented Generation (RAG) system that leverages the Gmail API to fetch email data and uses Mistral's latest LLM to generate context-aware responses. This project uses Qdrant for efficient vector similarity search to improve email retrieval performance.

## Project Structure
- **`main.py`**: Entry point for the application.
- **`config.py`**: Configuration for Qdrant, Mistral API, and embeddings.
- **`vector_store.py`**: Qdrant vector database operations and email storage.
- **`email_loader.py`**: Gmail API integration for fetching and indexing emails.
- **`rag.py`**: RAG pipeline with smart query routing and LLM integration.
- **`chat.py`**: Interactive chat interface.
- **`data_ingestion.py`**: Script for ingesting external data (CSV, JSON, text files).
- **`gmail_api.py`**: Gmail API helper functions.
- **`.env`**: Stores your API keys and configuration.
- **`credentials.json`**: Stores your Gmail API credentials.
- **`requirements.txt`**: Lists the required Python packages.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/RAG-Gmail-Assistant.git
cd RAG-Gmail-Assistant
```

### 2. Install Dependencies
Make sure you have Python 3.x installed. To install the required libraries, run:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory and add your API keys:
```
MISTRAL_API_KEY=your_mistral_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here (optional if using local Qdrant)
```

### 4. Set Up Qdrant
You have two options:

#### Option A: Local Qdrant (Recommended for Development)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option B: Qdrant Cloud
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io)
2. Create a cluster and get your URL and API key
3. Add them to your `.env` file

### 5. Obtain Gmail API Credentials
You will need to create a Google Cloud project and obtain a `credentials.json` file to access the Gmail API. Follow these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Navigate to APIs & Services > Credentials
4. Click Create Credentials and select OAuth 2.0 Client IDs
5. Download the `credentials.json` file and place it in the root directory of the project

#### Add Test Users
1. In the Google Cloud Console, go to OAuth consent screen
2. If your app is still in "testing" mode, under Test users, add the email addresses of users (including your own) that you want to allow access to the app
3. Save the settings

### 6. Run the Application

#### Load Emails from Gmail
```bash
python main.py --load-only
```

#### Start Interactive Chat
```bash
python main.py
```

#### Rebuild Index (Clear and Re-index)
```bash
python main.py --rebuild
```

### 7. Data Ingestion

You can ingest external data into Qdrant using the `data_ingestion.py` script:

#### Ingest from CSV
```bash
python data_ingestion.py --csv emails.csv
```

#### Ingest from JSON
```bash
python data_ingestion.py --json emails.json
```

#### Ingest Plain Text
```bash
python data_ingestion.py --text document.txt
```

#### Batch Ingest from Directory
```bash
python data_ingestion.py --dir ./data --pattern "*.json"
```

#### View Collection Statistics
```bash
python data_ingestion.py --stats
```

### CSV/JSON Format for Data Ingestion

#### CSV Format
```csv
gmail_msg_id,sender,recipient,date,subject,body
msg_123,alice@example.com,bob@example.com,2024-01-01T10:00:00,Hello,This is the email body
```

#### JSON Format
```json
[
  {
    "gmail_msg_id": "msg_123",
    "sender": "alice@example.com",
    "recipient": "bob@example.com",
    "date": "2024-01-01T10:00:00",
    "subject": "Hello",
    "body": "This is the email body"
  }
]
```

## Architecture

### Vector Search Pipeline
1. **Query Intent Detection**: Identifies if the query is temporal, contact-based, or semantic
2. **Smart Routing**: Routes to appropriate search strategy:
   - Temporal + Contact → Metadata sort
   - Contact → Hybrid search (filter + re-rank)
   - Semantic → Pure vector search
3. **Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
4. **Storage**: Qdrant with cosine similarity

### LLM Integration
- **Model**: Mistral Large (latest)
- **System Prompt**: French-language email analysis with strict grounding rules
- **Context**: Up to 5500 tokens of email context
- **Response**: Grounded in provided emails only

## User-Specific Files
The following files are generated dynamically when the program runs:

- **`token.json`**: Stores OAuth tokens for Gmail API authentication
- **`last_checked.txt`**: Records the timestamp of the last email retrieval

These files are essential for the program to function correctly but will be generated when the application is executed for the first time.

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
