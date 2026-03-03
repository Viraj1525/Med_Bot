import streamlit as st
import os
from dotenv import load_dotenv

from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------ CONFIG ------------------
st.set_page_config(page_title="MediAI Doctor", page_icon="🩺", layout="wide")

# ------------------ LOAD ENV ------------------
load_dotenv()

index_name = "medicalbot"

# ------------------ EMBEDDINGS ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ PINECONE ------------------
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ------------------ LLM ------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500,
    groq_api_key=os.getenv("GROQ_API_KEY").strip()
)

# ------------------ PROMPT ------------------
prompt_template = ChatPromptTemplate.from_template("""
You are a professional medical AI assistant.
Use the retrieved context to answer the question.
If unsure, say that you don't know.
Keep the answer concise (max 5 sentences).

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {
        "context": docsearch.as_retriever(search_kwargs={"k": 3}),
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ------------------ UI ------------------

st.title("🩺 MediAI - Intelligent Medical Assistant")
st.caption("Powered by Groq + Pinecone + HuggingFace")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if user_input := st.chat_input("Ask your medical question..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical knowledge..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})