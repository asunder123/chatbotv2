# app.py
# Minimal Streamlit chatbot (document-grounded)

import streamlit as st
from ingestion.extractor import extract_text_from_file
from engine.engine import CaseIndex

st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("💬 Document Chatbot")

if "index" not in st.session_state:
    st.session_state.index = None
if "chat" not in st.session_state:
    st.session_state.chat = []

uploaded = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx", "md", "html", "csv", "json"]
)

if uploaded:
    raw_text = extract_text_from_file(uploaded)

    index = CaseIndex()
    index.build_from_text(raw_text)

    st.session_state.index = index
    st.session_state.chat = []

    st.success(f"Indexed {len(index.sentences)} sentences")

if st.session_state.index:
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    query = st.chat_input("Ask anything about the document…")

    if query:
        st.session_state.chat.append({"role": "user", "content": query})

        results = st.session_state.index.query(query)

        if results:
            answer = results[0]["answer"]
            evidence = results[0]["evidence"]

            st.session_state.chat.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("🔎 Evidence (verbatim from document)"):
                    st.write(evidence)
        else:
            with st.chat_message("assistant"):
                st.write(
                    "I couldn’t find a closely matching sentence in the document."
                )
