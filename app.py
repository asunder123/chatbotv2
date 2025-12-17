# app.py
# Document Chatbot
# - Document-type aware extraction
# - Strict similarity-based retrieval
# - Sentence stitching supported
# - No orchestration layers

import streamlit as st

from ingestion.extractor import extract_text_from_file
from engine import CaseIndex


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Document Chatbot",
    layout="wide"
)

st.title("💬 Document Chatbot (Similarity Driven)")
st.caption(
    "Answers are generated strictly from document similarity. "
    "Evidence is shown verbatim."
)


# ============================================================
# Session state
# ============================================================

st.session_state.setdefault("index", None)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("doc_name", None)


# ============================================================
# Document upload & indexing
# ============================================================

uploaded = st.file_uploader(
    "Upload a document",
    type=["txt", "md", "csv", "json", "html", "docx", "pdf", "xlsx"]
)

if uploaded:
    # Avoid re-indexing same file repeatedly
    if st.session_state.doc_name != uploaded.name:
        raw_text = extract_text_from_file(uploaded)

        st.subheader("🔍 Extracted text preview")
        if raw_text.strip():
            st.code(raw_text[:2000])
        else:
            st.warning("No readable text could be extracted from this document.")

        index = CaseIndex()
        index.build_from_text(raw_text)

        st.session_state.index = index
        st.session_state.chat = []
        st.session_state.doc_name = uploaded.name

        st.success(f"Indexed {len(index.sentences)} sentences from `{uploaded.name}`")

        # Diagnostics (very useful during tuning)
        with st.expander("📊 Index diagnostics"):
            st.write("Total sentences:", len(index.sentences))
            if index.sentences:
                st.write("Example sentence:")
                st.code(index.sentences[0])


# ============================================================
# Chat interface
# ============================================================

if st.session_state.index:

    # Render chat history
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    query = st.chat_input("Ask a question about the document…")

    if query:
        # Store user query
        st.session_state.chat.append({
            "role": "user",
            "content": query
        })

        # Query engine
        results = st.session_state.index.query(query)

        if not results:
            answer = (
                "No sentence in the document is sufficiently similar to this query.\n\n"
                "Try using more concrete terms that appear in the document."
            )
            evidence = None
            score = 0.0
            stitched_count = 0
        else:
            top = results[0]
            answer = top.get("answer", "")
            evidence = top.get("evidence")
            score = top.get("score", 0.0)
            stitched_count = top.get("stitched_count", 1)

        # Store assistant response
        st.session_state.chat.append({
            "role": "assistant",
            "content": answer
        })

        # Render assistant response
        with st.chat_message("assistant"):
            st.write(answer)

            if evidence:
                with st.expander("🔎 Evidence (verbatim from document)"):
                    st.write(evidence)

            st.caption(
                f"Similarity score: {score:.4f} | "
                f"Sentences stitched: {stitched_count}"
            )

else:
    st.info("Upload a document to begin querying.")
