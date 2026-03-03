# Med_Bot
# 🩺 MediAI – Intelligent Medical RAG Chatbot

An AI-powered Medical Question-Answering system built using:

- 🧠 Groq (LLaMA 3.3 70B)
- 📚 Pinecone Vector Database
- 🤗 HuggingFace Embeddings
- 🔎 Retrieval Augmented Generation (RAG)
- 📄 Medical PDF Knowledge Base

---

## 🚀 Overview

MediAI is a Retrieval-Augmented Generation (RAG) based medical chatbot that answers health-related queries using a custom medical document dataset.

Instead of hallucinating responses like traditional LLMs, MediAI:

1. Retrieves relevant information from medical PDFs
2. Converts them into embeddings
3. Stores them in Pinecone
4. Uses Groq LLM to generate accurate answers

This ensures:
- Higher factual accuracy
- Reduced hallucination
- Context-aware medical responses

---

## 🏗️ Architecture

User Question  
↓  
Retriever (Pinecone Vector Search)  
↓  
Relevant Medical Context  
↓  
Groq LLM (LLaMA 3.3 70B)  
↓  
Final Answer  

---

## 📂 Project Structure

Medical_Chatbot/
│
├── rag_pipeline.py        # Core RAG backend
├── app.py                 # Streamlit frontend (optional)
├── .env                   # API Keys
├── requirements.txt
└── Data/                  # Medical PDF files

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|------------|
| LLM | Groq (LLaMA 3.3 70B Versatile) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | Pinecone |
| Framework | LangChain |
| Frontend | Streamlit |
| Language | Python |

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

PINECONE_API_KEY=your_pinecone_key  
GROQ_API_KEY=your_groq_key  

---

## 📦 Installation

Install all dependencies:

```bash
pip install langchain
pip install langchain-groq
pip install langchain-pinecone
pip install langchain-huggingface
pip install sentence-transformers
pip install pinecone-client
pip install python-dotenv
pip install streamlit
