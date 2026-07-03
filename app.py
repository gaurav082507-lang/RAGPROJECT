import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG Chat | Gaurav Gupta",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS — PROFESSIONAL THEME
# =========================================================
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
    }

    /* Header banner */
    .rag-header {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3444;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .rag-header h1 {
        color: #f9fafb;
        font-size: 1.9rem;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    .rag-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111318;
        border-right: 1px solid #262b38;
    }
    .sidebar-card {
        background: #1a1d27;
        border: 1px solid #2a2f3d;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .sidebar-card h4 {
        color: #e5e7eb;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }
    .sidebar-card p, .sidebar-card li {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .author-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        margin-top: 1rem;
    }
    .author-card .name {
        color: #f9fafb;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }
    .author-card .role {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-bottom: 0.7rem;
    }
    .author-card a {
        display: inline-block;
        background: #0a66c2;
        color: white !important;
        text-decoration: none;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .author-card a:hover {
        background: #084d92;
    }

    /* Status badge */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.35);
    }
    .status-pending {
        background: rgba(234,179,8,0.15);
        color: #facc15;
        border: 1px solid rgba(234,179,8,0.35);
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 12px;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="rag-header">
    <h1>📄 RAG Chat — Document Q&amp;A System</h1>
    <p>Upload a PDF and ask questions grounded strictly in its content, powered by Mistral AI &amp; LangChain.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")

    if "chain" in st.session_state:
        st.markdown('<span class="status-pill status-ready">● Ready to chat</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-pending">● Waiting for PDF</span>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>⚙️ How it works</h4>
        <p>1. Upload a PDF document<br>
        2. The system chunks &amp; embeds it<br>
        3. Retrieval uses MMR search (k=5)<br>
        4. Mistral LLM answers strictly from context</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div class="author-card">
        <div class="name">Gaurav Gupta</div>
        <div class="role">Developer</div>
        <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank">🔗 View LinkedIn Profile</a>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# BUILD RAG CHAIN (uses uploaded PDF) — CORE LOGIC UNCHANGED
# =========================================================
@st.cache_resource(show_spinner=False)
def build_chain(pdf_path):
    data = PyPDFLoader(pdf_path)
    docs = data.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    docs_chunk = splitter.split_documents(documents=docs)
    embedding_model = MistralAIEmbeddings(model='mistral-embed-2312')
    vectorstore = Chroma.from_documents(
        documents=docs_chunk,
        embedding=embedding_model,
        persist_directory='RagProjectDataBase'
    )
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 5,
            'fetch_k': 20,
            'lambda_mult': 0.5
        }
    )
    LLM = ChatMistralAI(model='mistral-small-2603')
    template = ChatPromptTemplate.from_messages([
        (
            'system', """You are a helpful assistant that answers questions using ONLY the information provided in the context below. Follow these rules strictly:

1. Base your answer solely on the provided context. Do not use outside knowledge or make assumptions.
2. If the context does not contain enough information to answer the question, say: "I don't have enough information in the provided documents to answer this question."
3. Do not fabricate facts, sources, or details not present in the context.
4. When possible, cite which document/source your answer came from (e.g., [Source: filename.pdf, page 3]).
5. If multiple sources conflict, point out the discrepancy rather than picking one silently.
6. Keep answers concise and directly relevant to the question asked. Expand only if the user asks for detail.
7. If the question is ambiguous, ask a clarifying question instead of guessing.
"""
        ),
        ('human', """"
    Context:
    {retrieved_chunks}

    Question:
    {user_query}
     """
        )
    ]
    )

    def get_context(docs, query):
        context = "\n".join(
            [doc.page_content for doc in docs]
        )
        return {'retrieved_chunks': context, 'user_query': query}

    parser = StrOutputParser()

    chains = retriever | RunnableLambda(lambda x: get_context(x, query=x)) | template | LLM | parser
    # NOTE: query is bound at invoke-time below via a wrapper, matching original script's
    # per-query chain construction behavior.
    return retriever, template, LLM, parser, get_context


# =========================================================
# SESSION STATE INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# HANDLE PDF UPLOAD
# =========================================================
if uploaded_file is not None:
    if st.session_state.get("uploaded_filename") != uploaded_file.name:
        with st.spinner("📚 Processing PDF — chunking & embedding document..."):
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            retriever, template, LLM, parser, get_context = build_chain(temp_path)

            st.session_state.retriever = retriever
            st.session_state.template = template
            st.session_state.LLM = LLM
            st.session_state.parser = parser
            st.session_state.get_context = get_context
            st.session_state.chain = True
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.messages = []

        st.success(f"✅ '{uploaded_file.name}' processed successfully! You can start asking questions.")


# =========================================================
# CHAT INTERFACE
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "chain" not in st.session_state:
    st.info("👋 Upload a PDF from the sidebar to begin chatting with your document.")
else:
    user_query = st.chat_input("Ask a question about your document...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                retriever = st.session_state.retriever
                template = st.session_state.template
                LLM = st.session_state.LLM
                parser = st.session_state.parser
                get_context = st.session_state.get_context

                chains = retriever | RunnableLambda(lambda x: get_context(x, query=user_query)) | template | LLM | parser
                response = chains.invoke(user_query)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top: 3rem; padding: 1rem; color:#6b7280; font-size:0.82rem;">
    Built with ❤️ by <b style="color:#9ca3af;">Gaurav Gupta</b> ·
    <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank" style="color:#0a66c2; text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
