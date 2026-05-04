# Local Wiki-RAG Assistant

A fully local Retrieval-Augmented Generation (RAG) system that answers questions about famous people and places using data scraped from Wikipedia. **No external APIs** -- everything runs on localhost using Ollama and ChromaDB.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Local Model](#running-the-local-model)
- [Ingesting Data](#ingesting-data)
- [Starting the Application](#starting-the-application)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Architecture](#architecture)

---

## Features

- Scrapes and indexes 40 Wikipedia articles (20 famous people, 20 famous places)
- Local LLM inference via Ollama (llama3.2)
- Local embedding generation via Ollama (nomic-embed-text)
- ChromaDB vector database with metadata-based filtering (person/place)
- Intelligent query routing (person, place, or mixed queries)
- Streaming responses (ChatGPT-like token-by-token output)
- Source citations with relevance scores
- Anti-hallucination guardrails (strict context grounding)
- Interactive CLI with `/sources`, `/stats`, `/help` commands

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Python 3.10 or higher** -- [Download Python](https://www.python.org/downloads/)
2. **Ollama** -- [Download Ollama](https://ollama.com/download)
3. **Git** (optional, for cloning the repository)

---

## Installation

### Step 1: Clone or download the project

```bash
cd path/to/your/directory
git clone <repository-url>
cd hw3_aiaided
```

Or simply download and extract the project files.

### Step 2: Install Python dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `wikipedia` -- Wikipedia article scraping
- `langchain` + `langchain-text-splitters` -- Text chunking
- `chromadb` -- Vector database
- `ollama` -- Local LLM and embedding client
- `rich` -- Beautiful CLI output formatting

---

## Running the Local Model

### Step 1: Install Ollama

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

After installation, verify it's working:

```bash
ollama --version
```

### Step 2: Pull the required models

You need two models: one for text generation and one for embeddings.

```bash
# Pull the LLM model (used for generating answers)
ollama pull llama3.2

# Pull the embedding model (used for vector search)
ollama pull nomic-embed-text
```

### Step 3: Verify models are available

```bash
ollama list
```

You should see both `llama3.2` and `nomic-embed-text` in the output.

> **Note:** Ollama must be running in the background whenever you use the application.
> On Windows, Ollama runs as a system tray application automatically after installation.
> If it's not running, start it from the Start Menu or run `ollama serve` in a terminal.

---

## Ingesting Data

Before you can ask questions, you need to scrape Wikipedia and build the vector database.

```bash
python main.py ingest
```

This command will:
1. Scrape 40 Wikipedia articles (20 people + 20 places)
2. Clean and chunk the text (800 characters per chunk with 150 character overlap)
3. Generate embeddings using `nomic-embed-text`
4. Store everything in a local ChromaDB database

**Expected output:**
```
Starting Wikipedia Ingestion
   Entities to process: 40
   Chunk size: 800 chars | Overlap: 150 chars

  + Albert Einstein -- 186 chunks
  + Marie Curie -- 95 chunks
  ...
  + Burj Khalifa -- 42 chunks

Ingestion complete! Total chunks: ~2500

Storing 2500 chunks in ChromaDB
   Embedding model: nomic-embed-text
   ...

Successfully stored 2500 chunks in ChromaDB!
```

> **Note:** The first run takes several minutes depending on your hardware (embedding generation is the bottleneck). Subsequent runs can be skipped unless you clear the database.

---

## Starting the Application

### Interactive Chat Mode

```bash
python main.py chat
```

This opens an interactive Q&A session. Type your question and press Enter. The response streams token-by-token like ChatGPT.

**In-chat commands:**
| Command | Description |
|---------|-------------|
| `/sources` | Show the source chunks used for the last answer |
| `/stats` | Display database statistics (total chunks, people, places) |
| `/help` | Show available commands |
| `/quit` | Exit chat mode |

### Clear Database

To delete the vector database and start fresh:

```bash
python main.py clear
```

### Show Usage

```bash
python main.py
```

---

## Example Queries

### People Queries
```
You > What did Marie Curie discover?
You > When was Albert Einstein born?
You > Compare Lionel Messi and Cristiano Ronaldo
You > What is Nikola Tesla famous for?
```

### Place Queries
```
You > Where is the Eiffel Tower located?
You > What was the Colosseum used for?
You > How tall is the Burj Khalifa?
You > When was the Taj Mahal built?
```

### Mixed / Cross-Domain Queries
```
You > Which person is associated with electricity?
You > Which famous place is located in Turkey?
```

### Failure Case (Anti-Hallucination)
```
You > Tell me about a random unknown person John Doe
Assistant > I'm sorry, I don't have information about that in my database...
```

### Viewing Sources
```
You > What did Marie Curie discover?
Assistant > [streaming response...]

You > /sources
  [1] Marie Curie (person) -- relevance: 0.892
      Marie Skłodowska Curie was a Polish and naturalized-French physicist...
  [2] Marie Curie (person) -- relevance: 0.845
      ...
```

---

## Project Structure

```
hw3_aiaided/
|-- main.py              # CLI entry point (ingest, chat, clear commands)
|-- ingest.py            # Wikipedia scraping and text chunking
|-- database.py          # ChromaDB initialization, embedding, and storage
|-- rag.py               # Query routing, retrieval, and LLM generation
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
|-- recommendation.md    # Production scaling recommendations
|-- product_PRD.md       # Product Requirements Document
+-- chroma_db/           # (auto-created) Persistent vector database storage
```

---

## Architecture

The system follows the **Option B** architecture: a single ChromaDB collection with metadata tags (`type: "person"` or `type: "place"`) to support cross-domain queries.

### Data Flow

```
Wikipedia Articles
       |
  [ingest.py] -- Scrape, clean, chunk (800 chars, 150 overlap)
       |
  [database.py] -- Embed via nomic-embed-text, store in ChromaDB
       |
  [rag.py] -- Classify query -> Retrieve top-K chunks -> Build prompt
       |
  [Ollama llama3.2] -- Generate grounded response (streaming)
       |
  [main.py] -- Display in interactive CLI
```

### Query Routing

1. **Entity matching**: Check if known entity names appear in the query
2. **Keyword classification**: Fall back to person/place keywords
3. **Metadata filtering**: Apply `type=person`, `type=place`, or no filter (mixed)

### Anti-Hallucination

The system prompt strictly instructs the LLM to:
- Answer **only** from retrieved context
- Say "I don't know" when the answer isn't available
- Cite sources at the end of each response

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ollama` not found | Make sure Ollama is installed and on your PATH |
| Connection refused | Ensure Ollama is running (`ollama serve`) |
| Slow embedding | First run is slow; subsequent queries use cached vectors |
| No data found | Run `python main.py ingest` first |
| Unicode errors | The application handles Windows encoding; update to latest version |
