# Production RAG LangChain

This project demonstrates a production-style Retrieval-Augmented Generation (RAG) application using LangChain and OpenAI.

## Features

- Document ingestion
- Text chunking
- Embeddings generation
- Vector database integration
- Semantic retrieval
- Conversational RAG
- FastAPI integration
- Hybrid search
- Session memory

## Tech Stack

- Python
- LangChain
- OpenAI
- ChromaDB
- FastAPI
- FAISS

## Project Structure

```bash
app/
data/
notebooks/
tests/
```

## Setup

Clone repository:

```bash
git clone https://github.com/vijaypingale/production-rag-langchain.git
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app/main.py
```

## Future Enhancements

- Hybrid retrieval
- Reranking
- Agentic workflows
- LangSmith observability
- AWS deployment