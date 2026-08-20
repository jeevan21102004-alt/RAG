# AdaptiveRAG

AdaptiveRAG is a beginner-friendly project for building an agentic RAG system step by step.

This step adds only a simple baseline RAG pipeline:

- load small sample documents
- split them into chunks
- create embeddings for each chunk
- save embeddings in a local JSON vector store
- retrieve the most relevant chunks for a user question

What is not included yet:

- answer generation with an LLM
- agents
- tool calling
- web search
- reinforcement learning

## Run the baseline query

Create and activate a virtual environment, install dependencies, then run:

```bash
python main.py --query "What is machine learning?"
```

You can also change the question or the number of retrieved chunks:

```bash
python main.py --query "What is overfitting?" --top-k 2
```

## How the pipeline works

1. `data_loader.py` reads the sample files in `data/`.
2. `chunking.py` splits each document into small overlapping text chunks.
3. `embeddings.py` turns the chunks into word-frequency embedding vectors.
4. `vector_store.py` stores those vectors and the chunk text in `storage/vector_store.json`.
5. `retrieval.py` embeds the user question, computes cosine similarity, and returns the best matches.
6. `app.py` prints the retrieved chunks and similarity scores.

## Project layout

- `data/` contains the sample machine-learning documents.
- `src/adaptive_rag/data_loader.py` loads documents from disk.
- `src/adaptive_rag/chunking.py` splits documents into chunks.
- `src/adaptive_rag/embeddings.py` creates the baseline embeddings.
- `src/adaptive_rag/vector_store.py` saves and loads the local vector database.
- `src/adaptive_rag/retrieval.py` ranks chunks by semantic similarity.
- `src/adaptive_rag/app.py` wires the pieces together and prints results.
- `main.py` is the root launcher for running the project from a checkout.

## Why this design

The code is split into small modules so each piece can be improved later without rewriting the whole project. That will make it easier to replace the simple baseline embedding model with a stronger retriever, then add agents and reinforcement learning in later steps.
