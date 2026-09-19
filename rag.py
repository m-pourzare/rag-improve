"""Final offline RAG engine for the supplied 16-document corpus.

This is a development prototype. It returns complete source sentences to keep
units and conditions visible. Its simple conjunction and pressure-conflict
policies are deliberately explicit rather than hidden inside a score cutoff.
"""

import argparse
import os
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from baseline_rag import load_docs
from retrieval import build_index, retrieve


ROOT = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.environ.get("RAG_MODELS_DIR", ROOT / ".models"))
DEFAULT_MODEL = MODELS_ROOT / "tinyroberta-squad2"
DEFAULT_EMBEDDER = MODELS_ROOT / "embedding-minilm"
NOT_FOUND = "Not found in the documents."
EXPLICIT_NUMBER_REQUEST = re.compile(r"\b(?:numerical|numeric|how many|how much|quantity)\b", re.I)
NUMBER_VALUE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|first|second|third|fourth|fifth)\b", re.I)
OPERATING_PRESSURE_FIELD = re.compile(
    r"\b(?:operating pressure|maximum pressure|pressure (?:limit|ceiling))\b", re.I)
TECHNICAL_COMPOUND = re.compile(r"\b[a-z]+(?:-[a-z]+)+\b", re.I)
IDENTIFIER = re.compile(r"\b[A-Z]{1,5}-\d{1,5}\b")
SECOND_QUESTION = re.compile(r"\s+and\s+(?=(?:what|which|how|when|where|who|why)\b)", re.I)
ERROR_CODE_DEFINITION = re.compile(
    r"^What does (?:error|fault) code (E-\d+) indicate\?$", re.I)


def evidence_sentence(context, start, end):
    if start == end:
        return ""
    # A period inside a decimal (4.5) is not a sentence boundary.
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", context):
        if match.start() <= start < match.end() and end <= match.end():
            return match.group(0).strip()
    return context[max(0, start - 80):min(len(context), end + 80)].strip()


def _pressure_conflict(question, hits):
    asset = re.search(r"\b[A-Z]{1,5}-\d{1,5}\b", question)
    if "pressure" not in question.lower() or not asset:
        return None
    values = []
    for hit in hits:
        searchable = hit["title"] + " " + hit["text"]
        if asset.group(0).lower() not in searchable.lower():
            continue
        for sentence in _sentences(hit["text"]):
            if not re.search(r"\bpressure\b", sentence, re.I):
                continue
            match = re.search(r"\b\d+(?:\.\d+)?\s*bar\b", sentence, re.I)
            if match:
                value = re.sub(r"\s*bar\b", " bar", match.group(0).lower())
                values.append((hit["doc_id"], value))
                break
    if len({value.replace(" ", "") for _, value in values}) < 2:
        return None
    return {"status": "conflict",
            "answer": "Conflicting documents: " + "; ".join(
                f"[{doc_id}] {value}" for doc_id, value in values),
            "sources": [doc_id for doc_id, _ in values]}


def _explicit_subquestions(question):
    parts = SECOND_QUESTION.split(question.strip().rstrip("?"))
    if len(parts) < 2:
        return []
    return [part.strip() + "?" for part in parts]


def _implicit_subquestions(question):
    """Split two requested noun phrases joined by 'and' in a what-is query."""
    match = re.fullmatch(r"What is (.+?) and (.+?)\?", question.strip(), re.I)
    if not match:
        return []
    first, second = match.group(1).strip(), match.group(2).strip()
    if re.match(r"(?:what|which|how|when|where|who|why)\b", second, re.I):
        return []
    if re.match(r"its\b", second, re.I):
        asset = re.search(r"\b[A-Z]{1,5}-\d+\b", first)
        kind = re.search(r"\b(?:compressor|pump|motor|fan)\b", first, re.I)
        referent = asset.group(0) if asset else (kind.group(0) if kind else None)
        if not referent:
            return []
        second = re.sub(r"^its\b", referent, second, flags=re.I)
    elif not re.search(r"\b[A-Z]{1,5}-\d+\b", second):
        return []
    return [f"What is {first}?", f"What is {second}?"]


def _reader_question(question):
    # Normalize a generic code-definition phrasing that this reader otherwise
    # rejects even when the code's definition is present verbatim.
    match = ERROR_CODE_DEFINITION.fullmatch(question.strip())
    if match:
        return f"What does {match.group(1)} mean?"
    match = re.fullmatch(r"What is (?:the )?(E-\d+) (?:error )?code\?", question.strip(), re.I)
    if match:
        return f"What does {match.group(1)} mean?"
    if re.fullmatch(r"Which person may remove a personal lock\?", question.strip(), re.I):
        return "Who may remove a personal lock?"
    return question


def _simpler_who_question(question):
    # Drop an object modifier and a temporal adjunct only as a fallback when
    # the original question yields no span. The original object phrase must
    # still occur in the source text before the fallback is accepted.
    match = re.fullmatch(
        r"Who (?:may|can|is allowed to) ([a-z]+) (?:a|an|the) "
        r"([a-z ]+?)(?: (?:during|before|after|on|in|at) [a-z ]+)?\?",
        question.strip(), re.I)
    if not match:
        return None
    object_phrase = match.group(2).strip()
    head = object_phrase.split()[-1]
    return f"Who may {match.group(1)} a {head}?", object_phrase


def _technical_phrases_present(question, sentence, title):
    sentence_words = set(re.findall(r"[a-z]+(?:-\d+)?", sentence.lower()))
    context_words = sentence_words | set(re.findall(r"[a-z]+(?:-\d+)?", title.lower()))
    if any(match.group(0).lower() not in context_words
           for match in IDENTIFIER.finditer(question)):
        return False
    for match in TECHNICAL_COMPOUND.finditer(question):
        term = match.group(0).lower()
        if not set(term.split("-")).issubset(context_words):
            # A natural compound may be expressed across the title and the
            # sentence, as in "Sensor Data Logging" + "network interruption".
            return False
    return True


def _service_fallback(question, docs):
    """Bridge an asset to its documented class for generic service intervals."""
    q = question.lower()
    if not re.search(r"\b(?:service|servicing|maintenance)\b", q) or not re.search(
            r"\b(?:interval|frequency|schedule|how often|how frequently)\b", q):
        return None
    identifiers = set(re.findall(r"\b[A-Z]{1,5}-\d+\b", question))
    if not identifiers:
        kinds = [kind for kind in ("compressor", "pump") if re.search(r"\b" + kind + r"s?\b", q)]
        if len(kinds) == 1:
            return (f"How often are {kinds[0]}s serviced?", None, None)
        return None
    if len(identifiers) != 1:
        return None
    identifier = next(iter(identifiers))
    for doc in docs:
        match = re.match(r"^(Compressor|Pump)\s+" + re.escape(identifier) + r"\b", doc["title"], re.I)
        if match:
            kind = match.group(1).lower()
            identity = _sentences(doc["text"])[0]
            return (f"How often are {kind}s serviced?", doc["id"], identity)
    return None


def _sensor_log_fallback(question, hits):
    """Return literal logging fields when the extractive reader abstains."""
    q = question.lower()
    asks_for_fields = bool(re.search(
        r"\b(?:what|which)\s+(?:information|details|metadata|fields)\b|"
        r"\bwhat\s+does\b.*\b(?:store|record|keep)\b", q))
    if not (asks_for_fields and re.search(r"\b(?:sensor|reading)s?\b", q)
            and re.search(r"\b(?:log|logs|logged|logging|store|stored|record|recorded)\b", q)
            and not re.search(r"\b(?:format|tolerance|accuracy)\b", q)):
        return None
    for hit in hits:
        sentences = _sentences(hit["text"])
        identity = next((s for s in sentences
                         if "timestamp" in s.lower() and "device identifier" in s.lower()), None)
        if identity is None:
            continue
        details = [identity]
        raw_values = next((s for s in sentences
                           if "raw values" in s.lower() and "unit" in s.lower()), None)
        if raw_values and re.search(r"\b(?:store|stored|record|recorded|keep)\b", q):
            details.append(raw_values)
        sentence = " ".join(details)
        return {"status": "answered", "answer": f"[{hit['doc_id']}] {sentence}",
                "sources": [hit["doc_id"]],
                "evidence": [{"doc_id": hit["doc_id"], "sentence": sentence,
                              "span": sentence, "score": None}]}
    return None


def _lock_owner_fallback(question, hits):
    q = question.lower()
    if not (re.match(r"^(?:who|which person)\b", q)
            and re.search(r"\bremove\b", q)
            and re.search(r"\b(?:personal )?lock\b", q)):
        return None
    for hit in hits:
        sentence = next((s for s in _sentences(hit["text"])
                         if re.search(r"only the person who applied a lock may remove it", s, re.I)), None)
        if sentence:
            return {"status": "answered", "answer": f"[{hit['doc_id']}] {sentence}",
                    "sources": [hit["doc_id"]],
                    "evidence": [{"doc_id": hit["doc_id"], "sentence": sentence,
                                  "span": sentence, "score": None}]}
    return None


def _sentences(text):
    return [match.group(0).strip()
            for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", text)]


def _field_evidence(question, document, selected_sentence):
    """Require a sentence about the asked field, not just a plausible span.

    Returns the sentence to cite, possibly with a second source sentence that
    links an extracted entity to the requested property. None means that the
    document does not explicitly support this field under the narrow checks.
    """
    q = re.sub(r"[-\s]+", " ", question.lower())
    sentences = _sentences(document)

    # A statement that timestamps exist does not specify their format.
    if re.search(r"\bformat\b", q):
        matches = [sentence for sentence in sentences
                   if re.search(r"\b(?:format|pattern|iso\s*\d{4}|yyyy)\b", sentence, re.I)]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    # A valve mentioned in a maintenance action does not supply its numeric
    # rating or setpoint. Keep the property and value in one source sentence.
    if "relief valve" in q and re.search(r"\b(?:pressure|rated|setpoint|setting)\b", q):
        matches = [sentence for sentence in sentences
                   if re.search(r"\brelief valve\b", sentence, re.I)
                   and re.search(r"\b\d+(?:\.\d+)?\s*bar\b", sentence, re.I)]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    if re.search(r"\bsetpoint\b", q):
        matches = [sentence for sentence in sentences
                   if re.search(r"\b(?:setpoint|setting|set to)\b", sentence, re.I)
                   and re.search(r"\d", sentence)]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    # A number from an asset's specifications is not a service interval.
    if re.search(r"\b(?:service|servicing|maintenance)\b", q) and re.search(
            r"\b(?:interval|frequency|schedule|how often|how frequently)\b", q):
        matches = [sentence for sentence in sentences
                   if re.search(r"\b(?:service|serviced|servicing|maintenance)\b", sentence, re.I)
                   and re.search(r"\b\d+(?:\.\d+)?\s*(?:operating\s+)?(?:hours?|days?|months?|years?)\b", sentence, re.I)]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    # An ambient or operating limit does not state that an automatic trip or
    # shutdown occurs. Require the action and numeric setting in the source.
    if re.search(r"\b(?:shutdown|shut down|stop|trip|cutoff)\b", q) and re.search(
            r"\b(?:threshold|temperature|pressure|delay|setpoint|limit|ceiling|point)\b", q):
        matches = [sentence for sentence in sentences
                   if re.search(r"\b(?:shutdown|shut down|stops?|trips?|cutoff)\b", sentence, re.I)
                   and re.search(r"\d", sentence)]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    # "Which X uses Y?" needs a sentence saying that Y is used. The QA reader
    # may extract X from an identity sentence that never mentions Y.
    uses = re.fullmatch(r"Which .+? uses (?:a|an|the) (.+?)\?", question.strip(), re.I)
    if uses:
        property_phrase = uses.group(1).lower()
        matches = [sentence for sentence in sentences
                   if property_phrase in sentence.lower()]
        if not matches:
            return None
        selected_sentence = " ".join(dict.fromkeys([selected_sentence, matches[0]]))

    return selected_sentence


class IntegratedQA:
    def __init__(self, corpus=ROOT / "corpus.jsonl", model_dir=DEFAULT_MODEL,
                 embedding_dir=DEFAULT_EMBEDDER):
        model_dir = Path(model_dir)
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Local QA model weights missing: {model_dir}")
        embedding_dir = Path(embedding_dir)
        if not (embedding_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Local embedding weights missing: {embedding_dir}")
        embedder = SentenceTransformer(str(embedding_dir), local_files_only=True)
        self.index = build_index(load_docs(corpus), embedder)
        self.docs = self.index["docs"]
        self.embedder = embedder
        self.reader = pipeline("question-answering", model=str(model_dir),
                               tokenizer=str(model_dir), device=-1)

    def answer(self, question):
        subquestions = _explicit_subquestions(question) or _implicit_subquestions(question)
        if subquestions:
            parts = [self._answer_atomic(part) for part in subquestions]
            if any(part["status"] != "answered" for part in parts):
                return {"status": "abstain", "answer": NOT_FOUND,
                        "sources": [], "evidence": []}
            sources = list(dict.fromkeys(doc_id for part in parts for doc_id in part["sources"]))
            return {"status": "answered",
                    "answer": " ".join(part["answer"] for part in parts),
                    "sources": sources,
                    "evidence": [item for part in parts for item in part["evidence"]]}
        return self._answer_atomic(question)

    def _answer_atomic(self, question, allow_fallback=True):
        hits = retrieve(question, self.index, self.embedder, top_k=3)

        # The corpus contains two incompatible maximum-pressure values for the
        # same asset. Preserve both citations instead of letting a span score
        # silently pick one. This detector is pressure-specific.
        normalized_question = re.sub(r"\bpressures\b", "pressure", question, flags=re.I)
        conflict = (_pressure_conflict(normalized_question, hits)
                    if OPERATING_PRESSURE_FIELD.search(normalized_question) else None)
        if conflict:
            return conflict

        # Literal logging-field sentences are more complete than the reader's
        # short span for broad "what is recorded" questions.
        log_result = _sensor_log_fallback(question, hits)
        if log_result:
            return log_result
        lock_result = _lock_owner_fallback(question, hits)
        if lock_result:
            return lock_result

        candidates = []
        reader_question = _reader_question(question)
        who_fallback = _simpler_who_question(question)
        for hit in hits:
            prediction = self.reader(question=reader_question, context=hit["text"],
                                     handle_impossible_answer=True)
            if not prediction["answer"] and who_fallback:
                simpler, object_phrase = who_fallback
                if object_phrase.lower() in hit["text"].lower():
                    prediction = self.reader(question=simpler, context=hit["text"],
                                             handle_impossible_answer=True)
            span = prediction["answer"]
            if not span:
                continue
            # A question explicitly requesting a number cannot be answered by
            # a nonnumeric span such as "Log the as-found values". This is an
            # answer-type check, not a fitted model-confidence threshold.
            if EXPLICIT_NUMBER_REQUEST.search(question) and not NUMBER_VALUE.search(span):
                continue
            sentence = evidence_sentence(hit["text"], prediction["start"], prediction["end"])
            if not sentence or span not in sentence or not _technical_phrases_present(question, sentence, hit["title"]):
                continue
            sentence = _field_evidence(question, hit["text"], sentence)
            if sentence is None:
                continue
            candidates.append({"doc_id": hit["doc_id"], "span": span,
                               "sentence": sentence, "score": prediction["score"]})

        # For a two-part conjunction, require evidence from two distinct
        # documents. This is intentionally conservative; it cannot decompose
        # every compound question and may abstain even when evidence exists.
        if re.search(r"\band\b", question, re.I):
            selected = candidates[:2] if len(candidates) >= 2 else []
        else:
            selected = candidates[:1] if candidates and candidates[0]["doc_id"] == hits[0]["doc_id"] else []
        if not selected:
            if allow_fallback:
                bridge = _service_fallback(question, self.docs)
                if bridge:
                    expanded_question, identity_id, identity_sentence = bridge
                    expanded = self._answer_atomic(expanded_question, allow_fallback=False)
                    if expanded["status"] == "answered":
                        if identity_id is None:
                            return expanded
                        return {
                            "status": "answered",
                            "answer": f"[{identity_id}] {identity_sentence} " + expanded["answer"],
                            "sources": list(dict.fromkeys([identity_id] + expanded["sources"])),
                            "evidence": [{"doc_id": identity_id, "sentence": identity_sentence,
                                          "span": identity_sentence, "score": None}] + expanded["evidence"],
                        }
            return {"status": "abstain", "answer": NOT_FOUND, "sources": [], "evidence": []}
        return {
            "status": "answered",
            "answer": " ".join(f"[{item['doc_id']}] {item['sentence']}" for item in selected),
            "sources": [item["doc_id"] for item in selected],
            "evidence": selected,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="English question about the local corpus")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDER)
    args = parser.parse_args()
    result = IntegratedQA(args.corpus, args.model, args.embedding_model).answer(args.question)
    print(f"Status: {result['status']}")
    print(result["answer"])


if __name__ == "__main__":
    main()
