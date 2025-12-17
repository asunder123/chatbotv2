# app.py
# Case Chatbot with multi-format document upload

import streamlit as st

from case_engine import *



st.set_page_config(page_title="Case Chatbot", layout="wide")
st.title("💬 Case Analysis Chatbot")

# ============================================================
# Session State
# ============================================================

if "case_index" not in st.session_state:
    st.session_state.case_index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# Sidebar: Upload
# ============================================================

with st.sidebar:
    st.header("📄 Upload Case")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["txt", "pdf", "docx", "md", "html", "csv", "json"]
    )

    chunk_size = st.slider("Chunk size", 60, 200, 120)
    overlap = st.slider("Chunk overlap", 10, 80, 30)

    if st.button("Load Document"):
        if uploaded_file:
            with st.spinner("Parsing and indexing document..."):
                text = extract_text_from_file(uploaded_file)
                st.session_state.case_index = build_case_index_from_text(
                    text, chunk_size, overlap
                )
                st.session_state.chat_history = []
            st.success("Document loaded. Start chatting.")
        else:
            st.warning("Please upload a document.")

# ============================================================
# Chat Interface
# ============================================================

if st.session_state.case_index is None:
    st.info("Upload a document to begin.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching evidence..."):
                results = st.session_state.case_index.query(
                    user_input,
                    top_k_chunks=3,
                    sentences_per_chunk=1
                )

            if not results:
                reply = "No explicit evidence found in the document."
                st.write(reply)
            else:
                reply = results[0]["answer"]
                st.write(reply)

                with st.expander("📌 Evidence"):
                    st.write(results[0]["source_chunk"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
