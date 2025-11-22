# Retrieval-Augmented QA Demo

Small end-to-end RAG demo that ingests a medical FAQ text file, builds a FAISS index with OpenAI embeddings, and serves an interactive Gradio chatbot to answer questions grounded in the ingested corpus. All orchestration lives in `rag_application/building_rag_application.py`.

- **Stack:** LangChain (RetrievalQA + OpenAI embeddings), FAISS, Gradio UI, `gdown` for dataset fetch, dotenv for config.
- **Data flow:** download text → clean/reshape into QA pairs → chunk → embed + persist FAISS index → run sample similarity checks → launch chat UI backed by the retriever.
- **Outputs:** chunked data and artifacts under `rag_application/output/` plus the FAISS index in `rag_application/faiss_doc_idx/`.

## Prerequisites

- Python 3.10+ recommended.
- An OpenAI API key set as `OPENAI_API_KEY` in `rag_application/.env`.
- (Optional) [`uv`](https://github.com/astral-sh/uv) installed if you want to use the script’s shebang (`#!/usr/bin/env -S uv run`); otherwise run with plain Python after installing deps.

## Setup

From the repo root:

1) Install dependencies:
   - With `uv`: `uv pip install -r rag_application/requirements.txt`
   - Or with pip/venv: `python -m venv .venv && source .venv/bin/activate && pip install -r rag_application/requirements.txt`
2) Create `rag_application/.env` containing `OPENAI_API_KEY=...`

## Run the pipeline + UI

Execute from the repo root so relative paths resolve:

```bash
uv run rag_application/building_rag_application.py
# or: python rag_application/building_rag_application.py
```

What happens:

- Ensures the dataset `input/ai-medical-chatbot.txt` exists, downloading via `gdown` if missing.
- Normalizes sections, builds QA pairs, and writes intermediates to `rag_application/output/sections.json` and `rag_application/output/qa_pairs.json`.
- Splits the corpus into overlapping chunks, embeds with OpenAI, and persists a FAISS index to `rag_application/faiss_doc_idx/`.
- Prints a couple of sample retrievals for sanity checks.
- Starts a Gradio chat interface (`DocumentQABot`) using a LangChain `RetrievalQA` chain with a custom prompt; the UI will open locally and can optionally generate a share link.

## Notes and tips

- Rerunning will reuse the downloaded dataset and existing FAISS index; delete `faiss_doc_idx/` if you want to rebuild embeddings from scratch.
- The UI prompt is in `PROMPT_TEMPLATE` inside `building_rag_application.py` if you want to tweak tone or safety guidance.
- For different datasets, swap the `DATASET_ID`/path constants and ensure chunk sizes still make sense for your content.
