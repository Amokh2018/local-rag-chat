import streamlit as st
from rag.rag_pipeline import run_rag

st.set_page_config(page_title="📄 Chat with Your Documents", layout="wide")
st.title("📄 Chat with Your Documents – Local RAG")

st.markdown("Ask a question and get answers directly from your local documents using a private RAG pipeline powered by local LLMs.")

question = st.text_input("❓ Enter your question:", placeholder="e.g., What is the main topic in this document?")

if question:
    with st.spinner("🔍 Retrieving and querying the LLM..."):
        answer, retrieved_chunks = run_rag(question)

        # Save for reuse or download
        st.session_state["retrieved_chunks"] = retrieved_chunks

        # Extract document names
        sources = [chunk['source'].split(" - Page")[0] for chunk in retrieved_chunks]
        unique_docs = sorted(set(sources))

        st.markdown("### 💬 Answer")
        st.success(answer)

        st.markdown("### 📚 References Used")
        for doc in unique_docs:
            st.markdown(f"- `{doc}`")

        st.markdown("### 🔎 Retrieved Chunks")
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            with st.expander(f"{idx}. Source: {chunk['source']}"):
                st.write(chunk['text'])
