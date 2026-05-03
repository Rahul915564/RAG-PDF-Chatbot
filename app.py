import streamlit as st
import os
import tempfile
from pathlib import Path

from document_processor import process_pdfs, get_vector_store
from chat_engine import get_answer

st.set_page_config(
    page_title="RAG Chatbot | PDF Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "language" not in st.session_state:
        st.session_state.language = "English"

initialize_session_state()

with st.sidebar:
    st.title("📚 PDF Documents")
    st.markdown("---")

    st.subheader("🌐 Response Language")
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button(
            "🇬🇧 English",
            use_container_width=True,
            type="primary" if st.session_state.language == "English" else "secondary",
        ):
            st.session_state.language = "English"
            st.rerun()
    with lang_col2:
        if st.button(
            "🇮🇳 हिंदी",
            use_container_width=True,
            type="primary" if st.session_state.language == "Hindi" else "secondary",
        ):
            st.session_state.language = "Hindi"
            st.rerun()

    if st.session_state.language == "Hindi":
        st.caption("✅ उत्तर हिंदी में दिए जाएंगे")
    else:
        st.caption("✅ Responses will be in English")

    st.markdown("---")

    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with",
    )

    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_files:
            if st.button("🔄 Process Documents", use_container_width=True, type="primary"):
                st.session_state.processing = True
                with st.spinner("Processing PDFs... This may take a moment."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            pdf_paths = []
                            for uploaded_file in uploaded_files:
                                path = Path(tmpdir) / uploaded_file.name
                                path.write_bytes(uploaded_file.read())
                                pdf_paths.append(str(path))

                            vector_store = process_pdfs(pdf_paths)
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            st.session_state.chat_history = []
                        st.success(f"✅ Processed {len(uploaded_files)} document(s)!")
                        st.session_state.processing = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {str(e)}")
                        st.session_state.processing = False
        else:
            st.info("✅ Documents already processed")

    st.markdown("---")

    if st.session_state.processed_files:
        st.subheader("📄 Loaded Documents")
        for fname in st.session_state.processed_files:
            st.markdown(f"• **{fname}**")
    else:
        st.info("No documents loaded yet.\nUpload PDFs to get started.")

    st.markdown("---")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8em; color: gray;'>
        <b>Powered by:</b><br>
        🦙 Groq (llama-3.1-8b-instant)<br>
        🤗 HuggingFace Embeddings<br>
        🗄️ ChromaDB<br>
        🦜 LangChain
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main area ──────────────────────────────────────────────────────────────────
is_hindi = st.session_state.language == "Hindi"

st.title("🤖 RAG Chatbot")
if is_hindi:
    st.markdown("*अपने PDF दस्तावेज़ों के बारे में हिंदी या अंग्रेजी में प्रश्न पूछें*")
else:
    st.markdown("*Ask questions about your uploaded PDF documents — in English or Hindi*")
st.markdown("---")

if not st.session_state.vector_store:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 3rem 0;'>
                <h2>👈 Get Started</h2>
                <p style='font-size: 1.1em;'>Upload your PDF documents in the sidebar, then start asking questions!</p>
                <br>
                <p><b>Features:</b></p>
                <p>📤 Upload multiple PDFs</p>
                <p>💬 Chat in English or Hindi / हिंदी या अंग्रेजी में चैट करें</p>
                <p>🌐 Language toggle in sidebar</p>
                <p>📌 Source citations for every answer</p>
                <p>🆓 100% Free — No paid API needed</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    if not st.session_state.chat_history:
        if is_hindi:
            st.info("💡 दस्तावेज़ लोड हो गए! अब कोई भी प्रश्न पूछें। उत्तर हिंदी में मिलेगा।")
        else:
            st.info("💡 Documents loaded! Ask me anything about them.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📌 Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(
                            f"**📄 {src['file']}** — Page {src['page']}\n\n"
                            f"> {src['snippet']}"
                        )

    placeholder = (
        "अपने दस्तावेज़ों के बारे में प्रश्न पूछें..."
        if is_hindi
        else "Ask a question about your documents..."
    )
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            spinner_text = "सोच रहा हूँ..." if is_hindi else "Thinking..."
            with st.spinner(spinner_text):
                try:
                    result = get_answer(
                        question=user_input,
                        vector_store=st.session_state.vector_store,
                        chat_history=st.session_state.chat_history[:-1],
                        language=st.session_state.language,
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📌 Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"**📄 {src['file']}** — Page {src['page']}\n\n"
                                    f"> {src['snippet']}"
                                )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    err_msg = f"❌ Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err_msg, "sources": []}
                    )
