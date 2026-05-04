# Product Requirements Document (PRD): Local Wikipedia RAG System

## 1. Project Overview
**Name:** Local Wiki-RAG Assistant  
**Goal:** Build a simplified, local ChatGPT-style system that answers questions about famous people and places using data ingested from Wikipedia. The system must run entirely on `localhost` without any external APIs. Search retrieves information, AI generates answers, and the system combines both into a complete AI application.

## 2. Technology Stack & Constraints
- **Language:** Python
- **LLM Engine:** Ollama (Local)
- **Generative Model:** `llama3.2` (or `mistral` / `phi3`)
- **Embedding Model:** `nomic-embed-text` (via Ollama)
- **Vector Database:** Chroma
- **Metadata/Relational DB:** SQLite
- **Interface:** CLI (Command Line Interface)
- **Strict Constraint:** NO external APIs (OpenAI, Anthropic, etc. are strictly forbidden). All inference and data storage must happen on localhost.

## 3. Core Requirements

### 3.1. Data Ingestion
The system must automatically scrape, clean, and store data from Wikipedia at least for the following **40 entities**. 

**Famous People (20 Total):**
1. Albert Einstein (Mandatory)
2. Marie Curie (Mandatory)
3. Leonardo da Vinci (Mandatory)
4. William Shakespeare (Mandatory)
5. Ada Lovelace (Mandatory)
6. Nikola Tesla (Mandatory)
7. Lionel Messi (Mandatory)
8. Cristiano Ronaldo (Mandatory)
9. Taylor Swift (Mandatory)
10. Frida Kahlo (Mandatory)
11. Isaac Newton
12. Galileo Galilei
13. Charles Darwin
14. Winston Churchill
15. Mahatma Gandhi
16. Nelson Mandela
17. Martin Luther King Jr.
18. Serena Williams
19. Michael Jordan
20. Muhammad Ali

**Famous Places (20 Total):**
1. Eiffel Tower (Mandatory)
2. Great Wall of China (Mandatory)
3. Taj Mahal (Mandatory)
4. Grand Canyon (Mandatory)
5. Machu Picchu (Mandatory)
6. Colosseum (Mandatory)
7. Hagia Sophia (Mandatory)
8. Statue of Liberty (Mandatory)
9. Pyramids of Giza (Mandatory)
10. Mount Everest (Mandatory)
11. Stonehenge
12. Acropolis of Athens
13. Petra
14. Chichen Itza
15. Angkor Wat
16. Mount Fuji
17. Niagara Falls
18. Victoria Falls
19. Sydney Opera House
20. Burj Khalifa

### 3.2. Data Chunking
- Documents will be large. Use a text splitter (e.g., RecursiveCharacterTextSplitter).
- **Strategy:** Fixed size chunks with overlap (e.g., 500-1000 characters with 10-20% overlap) to maintain context across paragraph breaks.

### 3.3. Embed and Store (Architecture Decision)
- **Choice:** **Option B - One vector store with metadata.**
- **Explanation/Justification:** We will use a single Chroma vector store where every chunk includes metadata: `{"type": "person"}` or `{"type": "place"}`. This is architecturally superior to Option A (two stores) because it easily supports cross-domain queries (e.g., "Compare Albert Einstein and the Eiffel Tower") without needing to query two separate databases and merge the results manually.

### 3.4. Retrieval Logic (Routing)
The system must parse the user query to determine the category before searching:
- **Routing Rules:** Implement a simple keyword or rule-based logic (e.g., checking if entity names or words like "who", "where" exist in the query).
- **Search Execution:** Retrieve the top-K most relevant chunks using similarity search. Filter by metadata (`type=person` or `type=place`) if the query is strictly about one category. If mixed, search the entire store.

### 3.5. Generation (RAG Prompting)
Construct a prompt injecting the retrieved context and the user query.
- **Strict Grounding:** The model MUST generate answers based *only* on the retrieved chunks.
- **Anti-Hallucination:** If the answer cannot be deduced from the retrieved context (e.g., "Who is the president of Mars?"), the model MUST explicitly reply: *"I don't know"* or *"The answer is not in my database."*
- **Source Display:** The system should optionally display the source chunks/metadata used to generate the answer.

### 3.6. User Interface (CLI)
Provide an interactive Command Line Interface. The CLI should have a menu or support commands to:
- **Ingest:** `python main.py ingest` (Starts the Wikipedia scraping, chunking, and embedding process).
- **Chat:** `python main.py chat` (Enters the interactive Q&A loop).
- **View Context:** A toggle or command within the chat (e.g., `/sources`) to print the retrieved chunks used for the last answer.
- **Clear:** `python main.py clear` (Deletes the local vector DB and resets the system).

## 4. Example Queries to Support
- **People:** "What did Marie Curie discover?", "Compare Lionel Messi and Cristiano Ronaldo"
- **Places:** "Where is the Eiffel Tower located?", "What was the Colosseum used for?"
- **Mixed:** "Which person is associated with electricity?", "Which famous place is located in Turkey?"
- **Failure Cases:** "Tell me about a random unknown person John Doe" -> System must fail gracefully and say it doesn't know.

## 5. Required Project Structure & Outputs
The codebase must be structured cleanly for a GitHub repository submission:
- `main.py`: Entry point for the CLI.
- `ingest.py`: Logic for Wikipedia API extraction and chunking.
- `database.py`: Logic for Chroma initialization, embedding, and storage.
- `rag.py`: Logic for query routing, retrieval, and LLM generation via Ollama.
- `requirements.txt`: All Python dependencies.
- `README.md`: Instructions for the instructor to install and run the code from scratch.
- `product_prd.md`: This document.
- `recommendation.md`: A separate file proposing how to scale this to production.

## 6. Optional Extensions (To Implement)
To ensure the highest evaluation grade, the agent should implement these targeted extensions:
- **Streaming Responses:** Stream the LLM output to the CLI like ChatGPT does, instead of waiting for the full response.
- **Citations:** Append `[Source: {Wikipedia Title}]` at the end of the generated answers.