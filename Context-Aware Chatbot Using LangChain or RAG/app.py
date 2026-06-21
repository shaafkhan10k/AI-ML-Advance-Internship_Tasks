"""
app.py - Context-Aware RAG Chatbot
====================================
A Streamlit chatbot that uses LangChain, FAISS, and Llama-3.1-8B-Instruct
to answer questions from a vectorized document store with conversation memory.

Usage:
    streamlit run app.py

Requirements:
    - FAISS index built (run build_vector_db.py first)
    - HF_TOKEN in .env file (parent directory)
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────
# Load Environment Variables
# ─────────────────────────────────────────────────────────────
# Look for .env in the parent directory (where the main project is)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found! Please set it in the .env file.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# Initialize Components (cached for performance)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    """Load the HuggingFace embedding model (cached across reruns)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_store(_embeddings):
    """Load the FAISS vector store from disk."""
    faiss_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    if not os.path.exists(faiss_path):
        st.error(
            "❌ FAISS index not found! Please run `python build_vector_db.py` first."
        )
        st.stop()

    return FAISS.load_local(
        faiss_path,
        _embeddings,
        allow_dangerous_deserialization=True
    )


def get_llm():
    """Create the LLM instance using HuggingFace Router (OpenAI-compatible)."""
    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.7,
        max_tokens=1024,
    )


def get_memory():
    """Create a new ConversationBufferMemory instance."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


def create_qa_chain(llm, retriever, memory):
    """Create the ConversationalRetrievalChain."""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )


# ─────────────────────────────────────────────────────────────
# Initialize Session State
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ─────────────────────────────────────────────────────────────
# Load Models and Create Chain
# ─────────────────────────────────────────────────────────────
embeddings = load_embeddings()
vector_store = load_vector_store(embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = get_llm()
qa_chain = create_qa_chain(llm, retriever, st.session_state.memory)


# ─────────────────────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1d29;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #9ca3af;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Source document chips */
    .source-chip {
        display: inline-block;
        background: linear-gradient(135deg, #1e3a5f, #2d5a87);
        color: #7dd3fc;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        margin: 2px 4px;
        border: 1px solid #2d5a87;
    }

    /* Sidebar info cards */
    .info-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .info-card h4 {
        color: #a78bfa;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    /* Status indicator */
    .status-online {
        color: #4ade80;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown(f'<span class="status-online">● Online</span>', unsafe_allow_html=True)

    st.divider()

    # Architecture info
    st.markdown("""
    <div class="info-card">
        <h4>⚙️ Architecture</h4>
        <p>
        <b>LLM:</b> Llama-3.1-8B-Instruct<br>
        <b>Embeddings:</b> all-MiniLM-L6-v2<br>
        <b>Vector Store:</b> FAISS<br>
        <b>Framework:</b> LangChain<br>
        <b>Memory:</b> ConversationBuffer
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Knowledge base info
    learned_facts_path = os.path.join(os.path.dirname(__file__), "data", "learned_facts.txt")
    has_learned = os.path.exists(learned_facts_path)
    learned_facts_html = "<br>📄 Learned Facts" if has_learned else ""

    st.markdown(f"""
    <div class="info-card">
        <h4>📚 Knowledge Base</h4>
        <p>
        📄 Artificial Intelligence<br>
        📄 Machine Learning<br>
        📄 Deep Learning<br>
        📄 Natural Language Processing<br>
        📄 Computer Vision{learned_facts_html}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = get_memory()
        # Recreate the chain with fresh memory
        qa_chain = create_qa_chain(llm, retriever, st.session_state.memory)
        st.rerun()

    # Conversation stats
    msg_count = len(st.session_state.chat_history)
    st.caption(f"💬 Messages: {msg_count}")

    st.divider()

    # Suggested questions
    st.markdown("#### 💡 Try Asking:")
    suggestions = [
        "What is deep learning?",
        "Compare ML and DL",
        "Explain transformers in NLP",
        "What are CNNs used for?",
        "How does transfer learning work?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=f"sug_{suggestion}", use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Context-Aware RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about AI, ML, Deep Learning, NLP & Computer Vision — powered by RAG + Conversation Memory</p>',
    unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                source_html = " ".join(
                    [f'<span class="source-chip">📄 {s}</span>' for s in sources]
                )
                st.markdown(
                    f"<div style='margin-top: 8px;'>{source_html}</div>",
                    unsafe_allow_html=True
                )


# ─────────────────────────────────────────────────────────────
# Handle User Input
# ─────────────────────────────────────────────────────────────
# Check for pending question from sidebar suggestions
user_input = None
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question

# Chat input
chat_input = st.chat_input("Ask a question about AI, ML, Deep Learning, NLP, or Computer Vision...")

if chat_input:
    user_input = chat_input

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Check if user wants the chatbot to remember a fact
    cleaned_input = user_input.strip()
    is_remember = False
    fact_to_remember = ""

    for prefix in ["remember:", "remember"]:
        if cleaned_input.lower().startswith(prefix):
            fact_to_remember = cleaned_input[len(prefix):].strip()
            is_remember = True
            break

    if is_remember and fact_to_remember:
        with st.chat_message("assistant"):
            with st.spinner("💾 Memorizing this fact..."):
                try:
                    # Write to physical file to persist against future vector DB rebuilds
                    learned_facts_path = os.path.join(os.path.dirname(__file__), "data", "learned_facts.txt")
                    os.makedirs(os.path.dirname(learned_facts_path), exist_ok=True)
                    with open(learned_facts_path, "a", encoding="utf-8") as f:
                        f.write(fact_to_remember + "\n\n")

                    # Add the fact dynamically to the in-memory FAISS database
                    vector_store.add_texts(
                        texts=[fact_to_remember],
                        metadatas=[{"source": learned_facts_path}]
                    )

                    # Save the updated FAISS database back to disk
                    faiss_path = os.path.join(os.path.dirname(__file__), "faiss_index")
                    vector_store.save_local(faiss_path)

                    # Success message
                    success_msg = f"✨ I have stored this in my database: *\"{fact_to_remember}\"* and will remember it for future chats."
                    st.markdown(success_msg)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": success_msg
                    })
                except Exception as e:
                    error_msg = f"❌ Error saving fact: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        st.rerun()
    else:
        # Generate response using LLM
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents & generating response..."):
                try:
                    # Recreate chain with current memory to ensure it's in sync
                    qa_chain = create_qa_chain(
                        llm, retriever, st.session_state.memory
                    )

                    result = qa_chain.invoke({"question": user_input})

                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    # Extract unique source file names
                    source_names = list(set(
                        os.path.basename(doc.metadata.get("source", "unknown"))
                        .replace(".txt", "")
                        .replace("_", " ")
                        .title()
                        for doc in source_docs
                    ))

                    # Display answer
                    st.markdown(answer)

                    # Display source chips
                    if source_names:
                        source_html = " ".join(
                            [f'<span class="source-chip">📄 {s}</span>' for s in source_names]
                        )
                        st.markdown(
                            f"<div style='margin-top: 8px;'>{source_html}</div>",
                            unsafe_allow_html=True
                        )

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_names
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })