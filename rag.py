"""
rag.py - Retrieval-Augmented Generation Module

Handles query routing, context retrieval from ChromaDB, and LLM generation
using Ollama's llama3.2 model. Implements strict grounding to prevent
hallucination and supports streaming responses.
"""

import sys
import os
import re

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import ollama
from database import query_collection, keyword_search_collection
from rich.console import Console

console = Console(force_terminal=True)

# ─── Configuration ─────────────────────────────────────────────────────────────

LLM_MODEL = "llama3.2"
TOP_K = 5  # Number of chunks to retrieve per entity/query

# ─── Entity Lists for Routing ─────────────────────────────────────────────────

PEOPLE_NAMES = [
    "albert einstein", "marie curie", "leonardo da vinci", "william shakespeare",
    "ada lovelace", "nikola tesla", "lionel messi", "cristiano ronaldo",
    "taylor swift", "frida kahlo", "isaac newton", "galileo galilei",
    "charles darwin", "winston churchill", "mahatma gandhi", "nelson mandela",
    "martin luther king", "serena williams", "michael jordan", "muhammad ali",
]

# Short/last names for better partial matching
PEOPLE_SHORT_NAMES = [
    "einstein", "curie", "da vinci", "shakespeare", "lovelace", "tesla",
    "messi", "ronaldo", "swift", "kahlo", "newton", "galileo", "darwin",
    "churchill", "gandhi", "mandela", "luther king", "serena", "jordan", "ali",
]

PLACE_NAMES = [
    "eiffel tower", "great wall of china", "taj mahal", "grand canyon",
    "machu picchu", "colosseum", "hagia sophia", "statue of liberty",
    "pyramids of giza", "mount everest", "stonehenge", "acropolis of athens",
    "petra", "chichen itza", "angkor wat", "mount fuji", "niagara falls",
    "victoria falls", "sydney opera house", "burj khalifa",
]

# Keyword indicators for query type detection
PERSON_KEYWORDS = ["who", "person", "scientist", "artist", "player", "athlete",
                    "singer", "writer", "inventor", "leader", "politician",
                    "born", "died", "discovered", "invented", "painted",
                    "he ", "she ", "his ", "her "]
PLACE_KEYWORDS = ["where", "place", "located", "location", "built", "tower",
                   "building", "monument", "mountain", "falls", "temple",
                   "wall", "canyon", "landmark", "visit", "height", "tall"]


def _check_entity_match(query_lower: str, full_names: list[str], short_names: list[str] | None = None) -> bool:
    """Check if any entity name (full or short) appears in the query."""
    if any(name in query_lower for name in full_names):
        return True
    if short_names and any(name in query_lower for name in short_names):
        return True
    return False


def classify_query(query: str) -> str:
    """
    Classify a user query into 'person', 'place', or 'mixed' to determine
    which metadata filter to apply during retrieval.

    Uses a two-pass approach:
    1. Check if any known entity names (full or partial) appear in the query.
    2. Fall back to keyword-based classification.

    Args:
        query: The user's natural language question.

    Returns:
        One of 'person', 'place', or 'mixed'.
    """
    query_lower = query.lower()

    has_person = _check_entity_match(query_lower, PEOPLE_NAMES, PEOPLE_SHORT_NAMES)
    has_place = _check_entity_match(query_lower, PLACE_NAMES)

    # If both entity types are mentioned, it's a mixed query
    if has_person and has_place:
        return "mixed"
    if has_person:
        return "person"
    if has_place:
        return "place"

    # Keyword-based fallback
    person_score = sum(1 for kw in PERSON_KEYWORDS if kw in query_lower)
    place_score = sum(1 for kw in PLACE_KEYWORDS if kw in query_lower)

    if person_score > place_score:
        return "person"
    elif place_score > person_score:
        return "place"

    # Default to mixed (search everything)
    return "mixed"


# ─── Entity Name Mapping ──────────────────────────────────────────────────────
# Maps short/partial names to the full canonical entity name used in the database

_ENTITY_NAME_MAP = {}
for _full in [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare",
    "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo",
    "Taylor Swift", "Frida Kahlo", "Isaac Newton", "Galileo Galilei",
    "Charles Darwin", "Winston Churchill", "Mahatma Gandhi", "Nelson Mandela",
    "Martin Luther King Jr.", "Serena Williams", "Michael Jordan", "Muhammad Ali",
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon",
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty",
    "Pyramids of Giza", "Mount Everest", "Stonehenge", "Acropolis of Athens",
    "Petra", "Chichen Itza", "Angkor Wat", "Mount Fuji", "Niagara Falls",
    "Victoria Falls", "Sydney Opera House", "Burj Khalifa",
]:
    _ENTITY_NAME_MAP[_full.lower()] = _full
    # Also add last name / short forms
    parts = _full.lower().split()
    if len(parts) >= 2:
        _ENTITY_NAME_MAP[parts[-1]] = _full  # Last name
        if len(parts) >= 3:
            _ENTITY_NAME_MAP[" ".join(parts[1:])] = _full


def _detect_entities_in_query(query: str) -> list[str]:
    """
    Detect which specific entities are mentioned in the query.
    Returns a list of canonical entity names found.
    """
    query_lower = query.lower()
    found = set()

    # Check full names first (longer matches take priority)
    for key, canonical in sorted(_ENTITY_NAME_MAP.items(), key=lambda x: -len(x[0])):
        if key in query_lower:
            found.add(canonical)

    return list(found)


def _extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a query for keyword-based search.
    Filters out common stop words and short words.
    Returns keywords in multiple case variants since ChromaDB $contains
    is case-sensitive.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
        "where", "when", "how", "does", "did", "do", "can", "could", "would",
        "should", "about", "with", "from", "for", "that", "this", "and", "or",
        "but", "not", "has", "have", "had", "been", "being", "will", "shall",
        "may", "might", "must", "its", "it", "they", "them", "their", "there",
        "here", "some", "any", "all", "most", "many", "much", "very", "more",
        "than", "then", "also", "just", "only", "into", "over", "upon", "like",
        "tell", "me", "us", "you", "your", "my", "our", "famous", "compare",
        "located", "associated", "person", "place", "people", "known",
    }
    # Extract words preserving original case
    words = re.findall(r'[a-zA-Z]+', query)
    # Filter by stop words (case-insensitive) and length
    filtered = [w for w in words if w.lower() not in stop_words and len(w) >= 3]
    # Return both original case and title case variants for case-sensitive matching
    keywords = set()
    for w in filtered:
        keywords.add(w)              # Original case from query
        keywords.add(w.title())      # Title case (e.g., "Turkey")
        keywords.add(w.lower())      # Lowercase
    return list(keywords)


def _retrieve_for_single_entity(query: str, entity_name: str, n_results: int = TOP_K) -> dict:
    """
    Retrieve chunks for a single specific entity.
    """
    return query_collection(
        query_text=query,
        n_results=n_results,
        entity_names=[entity_name],
    )


def _merge_results(*result_sets) -> tuple[list[str], list[dict], list[float]]:
    """
    Merge multiple ChromaDB result sets, deduplicating by document ID.
    Returns merged (documents, metadatas, distances).
    """
    seen_ids = set()
    all_docs = []
    all_metas = []
    all_dists = []

    for results in result_sets:
        if not results.get("documents") or not results["documents"][0]:
            continue
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
                all_metas.append(meta)
                all_dists.append(dist)

    return all_docs, all_metas, all_dists


def retrieve_context(query: str, query_type: str) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks from ChromaDB using a hybrid strategy:

    1. Entity-specific: If named entities are detected, query each one
       separately and merge results for balanced coverage.
    2. Hybrid fallback: For generic queries (no entities detected), combine
       vector similarity search with keyword-based document search.

    Args:
        query: The user's question.
        query_type: 'person', 'place', or 'mixed'.

    Returns:
        A tuple of (formatted_context_string, list_of_source_metadata).
    """
    mentioned_entities = _detect_entities_in_query(query)
    filter_type = query_type if query_type != "mixed" else None

    documents = []
    metadatas = []
    distances = []

    if len(mentioned_entities) >= 2:
        # ─── Multi-entity comparison: query each entity separately ─────
        # This ensures balanced coverage (e.g., both Einstein AND Tesla)
        per_entity = max(3, TOP_K // len(mentioned_entities) + 1)
        result_sets = []
        for entity in mentioned_entities:
            result = _retrieve_for_single_entity(query, entity, n_results=per_entity)
            result_sets.append(result)
        documents, metadatas, distances = _merge_results(*result_sets)

    elif len(mentioned_entities) == 1:
        # ─── Single entity: filter by that entity ─────────────────────
        results = _retrieve_for_single_entity(query, mentioned_entities[0], n_results=TOP_K)
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

    else:
        # ─── Generic query: hybrid retrieval ──────────────────────────
        # Step A: Keyword-based search for terms the embeddings might miss
        # These are high-precision results (exact text match) so they get priority
        keywords = _extract_keywords(query)
        keyword_results = keyword_search_collection(
            keywords=keywords,
            n_results=3,
            filter_type=filter_type,
        )

        # Step B: Vector similarity search
        vector_results = query_collection(
            query_text=query,
            n_results=TOP_K * 2,
            filter_type=filter_type,
        )

        # Merge: keyword matches first (high precision), then vector results
        seen_ids = set()

        # Add keyword results first -- these are exact text matches the
        # embeddings missed, so they should always be included
        if keyword_results["ids"]:
            for doc_id, doc, meta in zip(
                keyword_results["ids"],
                keyword_results["documents"],
                keyword_results["metadatas"],
            ):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    documents.append(doc)
                    metadatas.append(meta)
                    distances.append(0.15)  # High relevance for keyword matches

        # Fill remaining slots with vector similarity results
        if vector_results["documents"] and vector_results["documents"][0]:
            for doc_id, doc, meta, dist in zip(
                vector_results["ids"][0],
                vector_results["documents"][0],
                vector_results["metadatas"][0],
                vector_results["distances"][0],
            ):
                if doc_id not in seen_ids and len(documents) < TOP_K:
                    seen_ids.add(doc_id)
                    documents.append(doc)
                    metadatas.append(meta)
                    distances.append(dist)

    if not documents:
        return "", []

    # Build context string with source labels
    context_parts = []
    sources = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = 1 - dist  # Convert distance to similarity (cosine)
        context_parts.append(
            f"[Source {i + 1}: {meta['entity']} ({meta['type']}), Relevance: {similarity:.2f}]\n{doc}"
        )
        sources.append({
            "entity": meta["entity"],
            "type": meta["type"],
            "relevance": round(similarity, 3),
            "chunk_preview": doc[:100] + "..." if len(doc) > 100 else doc,
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def build_prompt(query: str, context: str) -> str:
    """
    Construct the RAG prompt with strict grounding instructions.

    The prompt enforces:
    - Answer ONLY from the provided context
    - Explicit "I don't know" if the answer isn't in context
    - Source citations

    Args:
        query: The user's question.
        context: The formatted context string from retrieved chunks.

    Returns:
        The full prompt string to send to the LLM.
    """
    if not context:
        return f"""You are a helpful assistant. The user asked the following question, but NO relevant information was found in the database.

Question: {query}

You MUST respond with: "I'm sorry, I don't have information about that in my database. I can only answer questions about the famous people and places stored in my knowledge base."

Do NOT make up an answer. Do NOT use your own knowledge."""

    return f"""You are a helpful and knowledgeable assistant. Your task is to answer the user's question based ONLY on the provided context from Wikipedia articles.

STRICT RULES:
1. Answer the question using ONLY the information provided in the context below.
2. If the answer cannot be found in the context, say: "I don't have enough information in my database to answer that question."
3. Do NOT use any external knowledge or make up facts.
4. Be concise and informative.
5. At the end of your answer, cite the source(s) you used in this format: [Source: Entity Name]
6. For comparison questions: If the context contains information about multiple entities, synthesize the available facts to highlight similarities and differences. You do NOT need an explicit comparison in the text -- use the facts provided about each entity to construct your own comparison.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def generate_response(query: str, stream: bool = True) -> tuple[str, list[dict], str]:
    """
    Full RAG pipeline: classify -> retrieve -> generate.

    Args:
        query: The user's question.
        stream: If True, stream the response token by token (prints to console).

    Returns:
        A tuple of (full_response_text, source_metadata_list, query_type).
    """
    # Step 1: Classify the query
    query_type = classify_query(query)

    # Step 2: Retrieve relevant context
    context, sources = retrieve_context(query, query_type)

    # Step 3: Build the prompt
    prompt = build_prompt(query, context)

    # Step 4: Generate response via Ollama
    full_response = ""

    if stream:
        stream_response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream_response:
            token = chunk["message"]["content"]
            full_response += token
            console.print(token, end="", style="bold white")
        console.print()  # Newline after streaming
    else:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        full_response = response["message"]["content"]
        console.print(full_response, style="bold white")

    return full_response, sources, query_type


if __name__ == "__main__":
    test_query = "What did Marie Curie discover?"
    print(f"\nQuery: {test_query}")
    print(f"Classification: {classify_query(test_query)}")
