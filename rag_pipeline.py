import os
from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# -------------------- DOCUMENT LOADING --------------------
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(data_path: str):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(documents)

    return text_chunks


# -------------------- EMBEDDINGS --------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------- PINECONE --------------------
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone

INDEX_NAME = "medicalbot"

pc = PineconeClient(api_key=PINECONE_API_KEY)

# Create index if not exists
existing_indexes = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


def create_vector_store(text_chunks):
    return Pinecone.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )


def load_existing_vector_store():
    return Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )


# -------------------- LLM (GROQ) --------------------
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500,
    groq_api_key=GROQ_API_KEY.strip()
)


# -------------------- RAG CHAIN --------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


prompt = ChatPromptTemplate.from_template("""
You are a professional medical AI assistant.
Use the provided context to answer the question.
If you don't know the answer, say that you don't know.
Keep the response concise (maximum 5 sentences).

Context:
{context}

Question:
{question}
""")


def create_rag_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# -------------------- MAIN FUNCTION --------------------
def initialize_rag(data_path=None):
    """
    If data_path is provided → creates index from PDFs
    If not provided → loads existing index
    """

    if data_path:
        text_chunks = load_and_split_documents(data_path)
        vector_store = create_vector_store(text_chunks)
    else:
        vector_store = load_existing_vector_store()

    rag_chain = create_rag_chain(vector_store)

    return rag_chain
