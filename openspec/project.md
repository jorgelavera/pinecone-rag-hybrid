# Project: pinecone-rag-hybrid

## Purpose
Scalable cloud RAG system with Pinecone: ingest pipeline (PDF/Markdown/JSON -> chunks with metadata; original text in Pinecone metadata), hybrid retrieval (BM25Retriever + Pinecone vector search combined via EnsembleRetriever) wrapped in RAGSystem returning top-5, evaluation via Precision@5/Recall@5 on hand-made Golden Set (multiple relevant docs per question), init script to verify/create Serverless index, README with replication steps, .env with required keys.

## Tech Stack
- Python 3.12
- LangChain (RecursiveCharacterTextSplitter, BM25Retriever, EnsembleRetriever)
- Pinecone (Serverless)
- OpenAI embeddings: text-embedding-3-small (dimension 1536)
- pytest for testing
- uv for venv/package management

## Conventions
- English for code, README, comments. Spanish only if explicitly mandated by assignment (field names not stated as requiring Spanish).
- Chunking target ~500-800 tokens via RecursiveCharacterTextSplitter.
- Store original text in Pinecone metadata (avoid extra DB lookups).
- Hybrid retriever: BM25 + Pinecone vector search via EnsembleRetriever; wrapped in RAGSystem returning top-5.
- Evaluation: Precision@5, Recall@5 on Golden Set with multiple relevant documents per question.

## Constraints
- Windows/PowerShell 5.1; use uv venv --python 3.12 .venv; install with uv pip install --python .venv\Scripts\python.exe -r requirements.txt
- Never use python from MS Store stub
- Parent workspace has mixed git repo with unrelated files - do not touch outside project dir
- No git repo/commits in this init; orchestrator handles
