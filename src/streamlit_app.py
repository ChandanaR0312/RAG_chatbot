import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import shutil

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Page config ----------
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #E0E7FF;
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }
    .doc-pill {
        background: #EEF2FF;
        color: #4338CA;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 2px 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind AI</h1>
    <p>Ask questions about your documents — powered by RAG (LangChain + ChromaDB + Groq)</p>
</div>
""", unsafe_allow_html=True)

# ---------- Core functions ----------
def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(chunks, embedding_model):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def format_history(messages, max_turns=3):
    recent = messages[-(max_turns * 2):] if len(messages) > max_turns * 2 else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "No previous conversation."


def get_answer(vector_store, question, chat_history):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    source_docs = retriever.invoke(question)
    context = format_docs(source_docs)
    history_text = format_history(chat_history)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions based only on the provided document context.
Do not infer or assume any information that is not explicitly stated — this includes gender, pronouns, personal details, or any other facts not directly written in the text.
Use the conversation history only to understand follow-up questions (e.g. "what about her education" referring to a previous answer), not as a source of facts.
If the answer is not in the context, say you don't know.

Conversation history:
{history}

Document context:
{context}

Question: {question}

Answer:"""
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question, "history": history_text})
    return answer, source_docs


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------- Sidebar: Document Manager ----------
with st.sidebar:
    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join(DATA_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s).")

    st.markdown("**Documents in knowledge base:**")
    existing_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if existing_pdfs:
        for pdf in existing_pdfs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<div class='doc-pill'>📄 {pdf}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{pdf}"):
                    os.remove(os.path.join(DATA_DIR, pdf))
                    st.rerun()
    else:
        st.caption("No documents uploaded yet.")

    st.divider()

    if st.button("⚙️ Process Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            docs = load_documents()
            if not docs:
                st.error("No PDFs found. Please upload at least one.")
            else:
                chunks = split_documents(docs)
                embedding_model = get_embedding_model()
                st.session_state.vector_store = build_vector_store(chunks, embedding_model)
                st.session_state.messages = []
                st.success(f"Processed {len(docs)} page(s) → {len(chunks)} chunks.")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.vector_store is not None:
        st.markdown("✅ **Knowledge base ready**")
    else:
        st.markdown("⚠️ **Not processed yet**")

# ---------- Main chat area ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 View sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}: {src} (Page {page})**")
                    st.caption(doc.page_content[:300] + "...")

if st.session_state.vector_store is None:
    st.info("👈 Upload PDF(s) and click **Process Documents** in the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_answer(
                    st.session_state.vector_store,
                    question,
                    st.session_state.messages[:-1]
                )
            st.markdown(answer)
            with st.expander("📚 View sources"):
                for i, doc in enumerate(sources, 1):
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}: {src} (Page {page})**")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})