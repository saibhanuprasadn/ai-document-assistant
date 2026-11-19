"""
AI Document Assistant — an end-to-end RAG chatbot web app.

Pipeline:
    upload -> extract text -> chunk (overlapping ~500-char chunks)
           -> TF-IDF index (scikit-learn) -> retrieve top-k chunks
           -> generate answer (Gemini if GEMINI_API_KEY is set,
              otherwise return the retrieved passages as an
              extractive answer, clearly labelled)

Named as "end-to-end" because every stage from document upload to a
sourced answer runs locally in this one app. Nothing is mocked: the
retrieval is real TF-IDF cosine similarity over the uploaded text.
"""

import os
import re
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Optional: FAISS vector index (HNSW) ----------------------------------
# The reel's point: a vector database does not scan every vector.
# It builds an *index* — here HNSW (Hierarchical Navigable Small World),
# a layered graph where search hops from neighbour to neighbour instead
# of comparing the question against every chunk. FAISS is the library
# that builds it; without faiss installed, retrieval falls back to
# brute-force cosine similarity over the same vectors.
try:
    import faiss
except ImportError:
    faiss = None

# ---- Optional: PDF extraction ------------------------------------------------
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled by requirements.txt
    PdfReader = None

# ---- Optional: Gemini generation --------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

# In-memory knowledge base: list of dicts {"text": str, "doc": str}
chunks = []
vectorizer = None
tfidf_matrix = None
faiss_index = None   # HNSW vector index over the TF-IDF vectors, if faiss is installed

CHUNK_SIZE = 500      # target characters per chunk
CHUNK_OVERLAP = 100   # characters carried over between chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def extract_text(filename, file_bytes):
    """Return the plain text of an uploaded .txt or .pdf file."""
    if filename.lower().endswith(".pdf"):
        if PdfReader is None:
            raise ValueError("PDF support (pypdf) is not installed.")
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # default: treat as plain text (utf-8, tolerant of bad bytes)
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of ~`size` chars.

    Splits on sentence boundaries where possible so chunks end
    mid-thought as rarely as possible; the overlap keeps context
    that spans a chunk boundary available to retrieval.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                out.append(current)
            # overlap: start the new chunk with the tail of the old one,
            # cut at a space so the chunk starts on a word boundary
            if current:
                tail = current[-overlap:]
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence
    if current:
        out.append(current)
    return out


def rebuild_index():
    """Rebuild the TF-IDF index from all stored chunks.

    The vectors go into two structures:
    * the TF-IDF matrix (always) — the source of truth for the vectors,
    * a FAISS HNSW index (only when faiss is installed) — the *vector
      index* the reel talks about: an approximate-nearest-neighbour
      graph that finds the closest chunks without scanning all of them.
      TF-IDF vectors are L2-normalised, so ranking by squared L2
      distance is the same ranking as cosine similarity, and the
      returned score is converted back to a cosine value.
    """
    global vectorizer, tfidf_matrix, faiss_index
    if not chunks:
        vectorizer, tfidf_matrix, faiss_index = None, None, None
        return
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    faiss_index = None
    if faiss is not None:
        try:
            dense = tfidf_matrix.astype("float32").toarray()
            dim = dense.shape[1]
            index = faiss.IndexHNSWFlat(dim, 32)  # M=32 links per node
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
            index.add(dense)
            faiss_index = index
        except Exception:
            faiss_index = None  # fall back to brute-force cosine


def add_document(filename, text):
    """Chunk a document's text, store the chunks, rebuild the index."""
    doc_chunks = chunk_text(text)
    for chunk in doc_chunks:
        chunks.append({"text": chunk, "doc": filename})
    rebuild_index()
    return len(doc_chunks)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve(question, top_k=3):
    """Return the top-k most similar chunks to the question.

    With a FAISS HNSW index, the question vector is looked up in the
    graph (fast, approximate). Without faiss, the same vectors are
    scanned with brute-force cosine similarity. Either way the
    returned ``score`` is the cosine similarity, so results are
    identical for a small index like this one.
    """
    if vectorizer is None or tfidf_matrix is None:
        return []
    q_vec = vectorizer.transform([question])
    if faiss_index is not None:
        q = q_vec.astype("float32").toarray()
        # HNSW returns squared L2 distances; for unit vectors
        # cos = 1 - d^2 / 2, so convert back to cosine scores.
        distances, ids = faiss_index.search(q, min(top_k, len(chunks)))
        out = []
        for d2, i in zip(distances[0], ids[0]):
            if i < 0:
                continue
            score = 1.0 - float(d2) / 2.0
            if score > 0:
                out.append({"text": chunks[i]["text"],
                            "doc": chunks[i]["doc"],
                            "score": round(score, 4)})
        return out
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        {"text": chunks[i]["text"], "doc": chunks[i]["doc"],
         "score": round(float(scores[i]), 4)}
        for i in ranked[:top_k] if scores[i] > 0
    ]


def generate_answer(question, retrieved):
    """Build an answer from retrieved chunks.

    If GEMINI_API_KEY is set and the google-generativeai package is
    available, ask Gemini to answer using ONLY the retrieved context.
    Otherwise return the retrieved passages verbatim, labelled as an
    extractive answer. Never invents content beyond the documents.
    """
    context = "\n\n".join(
        f"[Source: {r['doc']}] {r['text']}" for r in retrieved
    )
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Answer the question using ONLY the context below. "
                "If the context does not contain the answer, say so "
                "plainly instead of guessing.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = model.generate_content(prompt)
            return response.text.strip(), "generative (Gemini)"
        except Exception as exc:  # API/quota/network failure -> fall back
            return (
                "Gemini generation failed "
                f"({exc}); showing the retrieved passages instead.\n\n"
                + context,
                "extractive (Gemini fallback)",
            )
    return (
        "Extractive answer (no Gemini API key configured) — "
        "the most relevant passages from your documents:\n\n" + context,
        "extractive",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", has_docs=bool(chunks),
                           doc_count=len({c["doc"] for c in chunks}))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "documents_indexed":
                    len({c["doc"] for c in chunks}),
                    "chunks_indexed": len(chunks)})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request (use field 'file')."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    name = f.filename
    if not name.lower().endswith((".txt", ".pdf")):
        return jsonify({"error": "Only .txt and .pdf files are supported."}), 400
    try:
        text = extract_text(name, f.read())
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 400
    if not text.strip():
        return jsonify({"error": "No readable text found in the file."}), 400
    n = add_document(name, text)
    return jsonify({"status": "ok", "document": name,
                    "chunks_added": n, "total_chunks": len(chunks)})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Provide a 'question' in the JSON body."}), 400
    if not chunks:
        return jsonify({"error": "No documents uploaded yet. "
                                 "Upload a .txt or .pdf first."}), 400
    retrieved = retrieve(question, top_k=3)
    if not retrieved:
        return jsonify({
            "answer": "I could not find anything in your documents "
                      "that matches this question.",
            "mode": "no-match",
            "sources": [],
        })
    answer, mode = generate_answer(question, retrieved)
    return jsonify({
        "answer": answer,
        "mode": mode,
        "sources": [{"text": r["text"], "doc": r["doc"]}
                    for r in retrieved],
    })


if __name__ == "__main__":
    # Optional convenience: index the bundled sample document on startup
    # so the app is demo-ready immediately. Delete sample_docs/ to skip.
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    if os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".txt", ".pdf")):
                path = os.path.join(sample_dir, fname)
                with open(path, "rb") as fh:
                    try:
                        add_document(fname, extract_text(fname, fh.read()))
                    except Exception as exc:
                        print(f"Skipping sample {fname}: {exc}")
    print(f"Indexed {len(chunks)} chunks from sample documents.")
    app.run(host="127.0.0.1", port=5000, debug=False)
