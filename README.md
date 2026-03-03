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


🧠 How RAG Works in This Project
PDFs are loaded using DirectoryLoader
Text is split into chunks (500 size, 20 overlap)
Chunks are converted into embeddings
Stored inside Pinecone
Query retrieves top-k similar chunks
Groq LLM generates response using retrieved context

▶️ Running the Backend
First Time (Create Index from PDFs)
from rag_pipeline import initialize_rag
rag = initialize_rag("D:/Computer Science/Medical_Chatbot/Data")
response = rag.invoke("What are symptoms of dengue?")
print(response)

Later (Load Existing Index)
from rag_pipeline import initialize_rag
rag = initialize_rag()
response = rag.invoke("What is gigantism?")
print(response)

💻 Run Streamlit Frontend
streamlit run app.py


🏥 Example Queries
What are symptoms of dengue?
What is gigantism?
How is acne treated?
What causes high blood pressure?
What are complications of diabetes?

🔥 Features
✅ Retrieval-Augmented Generation
✅ Reduced hallucination
✅ Fast Groq inference
✅ Scalable vector search
✅ Custom medical knowledge base
✅ Modular backend architecture
✅ Deployment-ready structure

📈 Future Improvements
Streaming token-by-token responses
Conversational memory
Premium doctor dashboard UI
Multi-document upload
Hybrid search (BM25 + Vector)
Medical citation highlighting
Authentication system

⚠️ Disclaimer
This chatbot is for educational and research purposes only.
It does not replace professional medical advice.
Always consult a licensed healthcare provider.

👨‍💻 Author
Viraj Agrawal
AI/ML Developer | RAG Systems Builder
