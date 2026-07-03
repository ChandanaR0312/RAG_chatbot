# DocuMind AI — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of uploaded PDF documents. Built with LangChain, ChromaDB, Sentence Transformers, and Groq's LLM, with a Streamlit-based chat interface.

## Overview

This project lets users upload one or more PDF documents, processes them into a searchable knowledge base, and answers natural language questions using only the content of those documents. It includes:

- A **Streamlit web app** (`src/streamlit_app.py`) with a chat interface, multi-document upload, source citations, and conversation memory
- A **command-line version** (`src/app.py`) for quick local testing

### How it works
1. PDF documents are loaded and split into overlapping text chunks
2. Each chunk is converted into a vector embedding using `sentence-transformers/all-MiniLM-L6-v2`
3. Embeddings are stored in a local ChromaDB vector database
4. When a user asks a question, the most relevant chunks are retrieved
5. Retrieved chunks are passed as context to Groq's LLM (`llama-3.1-8b-instant`), which generates a grounded answer
6. The app displays the answer along with the source document/page it came from

## Tech Stack
- **Python 3.11**
- **LangChain** — orchestration
- **ChromaDB** — vector database
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — embeddings
- **Groq** (`llama-3.1-8b-instant`) — LLM for answer generation
- **Streamlit** — web interface

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChandanaR0312/RAG_chatbot.git
cd RAG_chatbot
```

### 2. Create and activate a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Running the Application

### Streamlit web app (recommended)
```bash
python -m streamlit run src/streamlit_app.py
```
Then open `http://localhost:8501` in your browser. Upload one or more PDFs, click **Process Documents**, and start asking questions.

### Command-line version
```bash
python src/app.py
```
Place PDF files in the `data/` folder before running. Type your questions when prompted; type `exit` to quit.

## Project Structure
RAG_chatbot/
├── data/                  # PDF documents (knowledge source)
├── src/
│   ├── app.py             # CLI version
│   └── streamlit_app.py   # Web app version
├── chroma_db/             # Persisted vector database (generated)
├── requirements.txt
├── .env                   # API keys (not committed)
└── README.md

## Features
- Multi-PDF upload and document management
- Chat interface with conversation history
- Source citations showing which document/page each answer came from
- Prevents the LLM from inferring unstated details (e.g., gender, personal facts) not present in the source documents