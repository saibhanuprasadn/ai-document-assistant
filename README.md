# AI Document Assistant

An end-to-end RAG (Retrieval-Augmented Generation) chatbot web app.
Upload your own documents (.txt / .pdf), ask questions in plain English,
and get answers grounded in those documents — with the exact source
passages shown next to every answer.

## Architecture

```
upload (.txt/.pdf)
   │  extract text (plain read / pypdf)
   ▼
 chunk  — ~500-char chunks, ~100-char overlap, split on sentence
          boundaries so chunks stay readable
   │  each chunk remembers its source document
   ▼
 index  — TF-IDF vectors (scikit-learn TfidfVectorizer, English
          stop-words removed); one vector per chunk. The vectors are
          then put in a FAISS **HNSW vector index** (the "index" in a
          vector database): a layered graph that finds the nearest
          chunks by hopping between neighbours instead of scanning
          every chunk. Without faiss installed, retrieval falls back
          to brute-force cosine similarity over the same vectors,
          with identical results on a small index.
   │  index is rebuilt after every upload
   ▼
 retrieve — the question is vectorised the same way; cosine
            similarity ranks all chunks; top 3 are returned
   │  (TF-IDF vectors are L2-normalised, so a dot product
   │   is the cosine similarity)
   ▼
 generate — if GEMINI_API_KEY is set, Gemini (gemini-1.5-flash)
            answers using ONLY the retrieved chunks, and says
            it doesn't know if the answer isn't in them.
            Otherwise the app returns the retrieved passages
            verbatim as a clearly-labelled extractive answer.
            No answer is ever invented beyond the documents.
```

Everything runs locally (in-memory index, no database, no paid
service). Uploaded documents are indexed in memory; the index is
rebuilt on every upload. Nothing is deployed to the cloud.

## Install and run

```bash
cd ~/workspace/projects/ai-document-assistant

# 1. Create the virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Start the server (the bundled sample document about how RAG
#    works is indexed automatically on startup)
.venv/bin/python app.py

# 3. Open in a browser
#    http://127.0.0.1:5000
```

To enable Gemini-generated answers instead of extractive answers,
set an API key before starting the server (optional — the app works
fully without it):

```bash
export GEMINI_API_KEY=your_key_here
.venv/bin/python app.py
```

To stop the server: `Ctrl+C`, or `pkill -f 'app.py'` from another
terminal (run it from a script file, not from a shell whose own
command line contains the pattern).

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI (single page) |
| `/api/health` | GET | `{"status": "ok", "documents_indexed": N, "chunks_indexed": M}` |
| `/api/upload` | POST | Multipart form, field `file` (.txt or .pdf) |
| `/api/ask` | POST | JSON `{"question": "..."}` → `{"answer", "mode", "sources": [{"text", "doc"}]}` |

Example:

```bash
curl http://127.0.0.1:5000/api/health
curl -F "file=@sample_docs/how-rag-works.txt" http://127.0.0.1:5000/api/upload
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is chunking in RAG?"}' \
  http://127.0.0.1:5000/api/ask
```

## Example questions (sample document)

- "What is chunking in RAG?"
- "What are the advantages of RAG?"
- "What does TF-IDF do?"
- "What happens if the retrieved chunks do not contain the answer?"
- "How is RAG different from using a language model on its own?"

## Interview talking points

Say these in your own words — each is one short answer to a likely
question:

1. **What does it do end to end?** — "You upload a PDF or text file,
   the app extracts the text, splits it into overlapping chunks,
   builds a TF-IDF search index, and when you ask a question it
   retrieves the three most similar chunks and answers from those
   passages, showing the sources."
2. **Why chunk the documents?** — "A whole document is too long to
   search or feed to a model at once. I split it into roughly
   500-character chunks on sentence boundaries, with a 100-character
   overlap so a sentence that crosses a boundary isn't lost."
3. **How does retrieval work?** — "Each chunk becomes a TF-IDF
   vector, which weights distinctive words higher than common ones.
   The question is turned into a vector the same way, cosine
   similarity ranks the chunks, and the top three are used."
4. **What happens without an API key?** — "The app still works: it
   returns the retrieved passages as a clearly-labelled extractive
   answer. With a Gemini key set, the same passages are passed to
   the model as context and it writes a grounded answer instead."
5. **How do you stop it making things up?** — "The model is
   instructed to use only the retrieved context, and if the answer
   isn't in the documents the app says so plainly. Every answer
   shows its source chunks, so you can check it."
6. **What index does the vector database use?** — "I use a FAISS
   HNSW index — a layered graph that hops between neighbour vectors
   to find the closest chunks instead of scanning all of them. The
   vectors are TF-IDF, and because they're L2-normalised the graph's
   distance ranking is the same as cosine similarity, so the scores
   match brute force exactly. Without faiss installed the app falls
   back to brute-force cosine."

## Files

- `app.py` — the whole app (Flask routes, ingestion, retrieval, generation)
- `templates/index.html` — the single-page web UI
- `sample_docs/how-rag-works.txt` — bundled demo document (original text)
- `requirements.txt` — Flask, scikit-learn, numpy, pypdf, google-generativeai (optional)

## Verified (11 Nov 2026)

Built and run on this machine with Python 3.12, venv `.venv`:

- `GET /api/health` → `{"chunks_indexed": 10, "documents_indexed": 1, "status": "ok"}`
- `POST /api/upload` (sample doc) → `{"chunks_added": 10, "status": "ok", "total_chunks": 20}`
- `POST /api/ask` "What is chunking in RAG?" → extractive answer with 3
  sources from `how-rag-works.txt` (ingestion/chunking passages)
- `POST /api/ask` "What are the advantages of RAG?" → extractive answer
  quoting the advantages paragraph (update by upload, source citations,
  private local index)
- `POST /api/ask` "What does TF-IDF do?" → extractive answer quoting
  the TF-IDF definition (term frequency-inverse document frequency,
  distinctive words weighted higher)

## Verified (12 Nov 2026) — FAISS HNSW vector index added

- `faiss-cpu==1.9.0` installed in `.venv`; HNSW index builds on
  startup (`ntotal: 10` chunks from the sample doc).
- Same three sample questions return extractive answers with 3
  sources each; top chunks and cosine scores are **identical** to
  brute-force cosine (e.g. chunking question: 0.1561, 0.1181, 0.1086
  both ways) — the index only changes the search, not the answers.
- `POST /api/upload` (sample doc) → 10 chunks added, 20 total;
  `GET /api/health` → 20 chunks, 1 document. Server stopped after
  the test. Without faiss installed the app still runs and falls
  back to brute-force cosine.
