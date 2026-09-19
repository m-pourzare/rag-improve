# Offline RAG

This project answers questions using the documents in `corpus.jsonl`.

It runs locally, finds relevant documents by comparing the meaning of the
question with each document, returns the supporting source, handles
multi-document questions and conflicting information, and returns
`Not found in the documents` when there is not enough evidence.

## Setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .models
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='.models/embedding-minilm'); snapshot_download('deepset/tinyroberta-squad2', local_dir='.models/tinyroberta-squad2')"
```

The models only need to be downloaded once. After that, the application runs
offline.

## Run

Pass a question to `rag.py`:

```bash
python rag.py "What is the rated output of C-100?"
```

Example output:

```text
Status: answered
[DOC-03] Rated output is 6 m3/min at 8 bar.
```

## Evaluation

```bash
python evaluate.py --system baseline
python evaluate.py --system final --show-answers
```

`evaluation.csv` contains the 12 evaluation questions and their expected
sources. The final system passes 12 questions (100%). The supplied baseline
passes 7 questions (58.3%).
