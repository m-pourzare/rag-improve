import re
import numpy as np


IDENTIFIER = re.compile(r"\b[A-Z]{1,5}-\d{1,5}\b")


def _normalize(vectors):
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Embeddings must have nonzero finite norms")
    return vectors / norms


def build_index(docs, model):
    """Embed each short document once, using its title and text."""
    if not docs:
        raise ValueError("Corpus is empty")
    searchable = [f"{doc['title']}\n{doc['text']}" for doc in docs]
    vectors = _normalize(model.encode(searchable))
    return {"docs": docs, "texts": searchable, "vectors": vectors}


def retrieve(query, index, model, top_k=3):
    """Return the most similar documents, with exact identifiers first."""
    if not query.strip():
        raise ValueError("Query must not be empty")

    query_vector = _normalize(model.encode([query]))[0]
    scores = np.einsum("ij,j->i", index["vectors"], query_vector)

    identifiers = {match.group(0).lower() for match in IDENTIFIER.finditer(query)}

    def ranking_key(position):
        exact_match = bool(identifiers) and all(
            identifier in index["texts"][position].lower()
            for identifier in identifiers
        )
        return exact_match, float(scores[position])

    order = sorted(range(len(index["docs"])), key=ranking_key, reverse=True)[:top_k]
    return [
        {
            "doc_id": index["docs"][position]["id"],
            "title": index["docs"][position]["title"],
            "text": index["docs"][position]["text"],
            "score": float(scores[position]),
        }
        for position in order
    ]
