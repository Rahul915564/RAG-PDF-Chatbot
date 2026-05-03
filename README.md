# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, ChromaDB, HuggingFace Embeddings, and Groq.

## Features

- Upload multiple PDF documents
- Extract and chunk text from PDFs
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- Store vectors in ChromaDB
- Query using Groq API (llama3-8b-8192) — free tier
- Beautiful Streamlit UI with chat interface, source citations, and chat history
- Hindi and English support

## Setup

1. Get a free Groq API key at https://console.groq.com
2. Set `GROQ_API_KEY` environment variable
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo and set `GROQ_API_KEY` as a secret
4. Deploy!
