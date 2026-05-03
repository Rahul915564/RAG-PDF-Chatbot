# 🤖 RAG PDF Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)

A production-ready **Retrieval Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions in **English or Hindi**. Powered by Groq LLM, LangChain, and ChromaDB vector store.

🌐 **Live Demo:** [rag-pdf-chatbot-jnncvfjyccvr5beoufpt92.streamlit.app](https://rag-pdf-chatbot-jnncvfjyccvr5beoufpt92.streamlit.app)

---

## ✨ Features

- 📄 **Upload multiple PDFs** — supports up to 200MB per file
- 🔍 **Semantic Search** — finds relevant content using vector embeddings
- 🌐 **Bilingual Support** — ask questions in English or Hindi
- 📌 **Source Citations** — shows exactly which part of the PDF answered your question
- ⚡ **Fast Responses** — powered by Groq's ultra-fast LLM inference
- 🚀 **Deployed on Streamlit Cloud** — accessible from anywhere

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Groq (Llama 3) |
| **Orchestration** | LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | Sentence Transformers |
| **PDF Processing** | PyMuPDF, PyPDF2 |
| **Language** | Python 3.11 |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Streamlit UI  │  ← Upload PDFs, Select Language
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │  ← Extract text from PDFs
│ Processor       │  ← Split into chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Store    │  ← Generate embeddings
│ (ChromaDB)      │  ← Store & retrieve semantically
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq LLM       │  ← Generate answer with context
│  (Llama 3)      │  ← Support English & Hindi
└────────┬────────┘
         │
         ▼
    Answer + Source Citations
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Rahul915564/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY in .env file

# 4. Run the app
streamlit run app.py
```

### Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Project Structure

```
RAG-PDF-Chatbot/
├── app.py                  # Main Streamlit application
├── document_processor.py   # PDF processing & vector store logic
├── chat_engine.py          # LLM chat engine with LangChain
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── README.md
```

---

## 💡 How It Works

1. **Upload PDFs** — User uploads one or more PDF documents
2. **Text Extraction** — PyMuPDF extracts text from each page
3. **Chunking** — LangChain splits text into overlapping chunks
4. **Embeddings** — Sentence Transformers converts chunks to vectors
5. **Storage** — ChromaDB stores vectors for fast retrieval
6. **Query** — User asks a question in English or Hindi
7. **Retrieval** — ChromaDB finds the most relevant chunks
8. **Generation** — Groq LLM generates a contextual answer
9. **Citation** — App shows which PDF section was used

---

## 🔧 Key Dependencies

```
streamlit>=1.31.0
langchain>=0.2.0
langchain-groq>=0.1.0
chromadb==0.5.3
sentence-transformers>=2.7.0
PyPDF2>=3.0.0
pymupdf>=1.23.0
groq>=0.4.0
numpy==1.26.4
protobuf==3.20.3
```

---

## 🌟 What I Learned

- Building end-to-end RAG pipelines with LangChain
- Vector embeddings and semantic search with ChromaDB
- Deploying ML apps on Streamlit Cloud
- Resolving Python dependency conflicts in production
- Working with LLM APIs (Groq)

---

## 🤝 Connect

**Rahul** — [GitHub](https://github.com/Rahul915564) · [LinkedIn](https://linkedin.com/in/rahul915564)

---

⭐ **If you find this project useful, please give it a star!**
