import streamlit as st
import sys
sys.tracebacklimit=0
from index import PDFIndexer  
from query import RAGAgent
from pathlib import Path

indexer = PDFIndexer()
rag_agent = RAGAgent()

st.set_page_config(page_title="Legal Document Assistant", layout="centered")
st.title("📑 Legal Document Assistant")

if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing PDF..."):
        pdf_path = Path("uploaded") / uploaded_file.name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())


        paragraphs = indexer.extract_paragraphs(str(pdf_path))
        indexed = indexer.index_paragraphs(paragraphs, pdf_name=pdf_path.stem)

        if indexed:
            st.success(f"✅ Indexed {len(paragraphs)} paragraphs from {uploaded_file.name}")
            st.session_state.indexing_done = True
            st.session_state.pdf_name = pdf_path.stem
        else:
            st.error("❌ Indexing failed.")

if st.session_state.indexing_done:
    with st.form(key="query_form"):
        query = st.text_input("🔍 Ask a question about the document")
        submitted = st.form_submit_button("Search")
        
    if  st.session_state.indexing_done and submitted and query:
        with st.spinner("Retrieving answer..."):
            response = rag_agent.retrieve(
                query_text=query,
                pdf_name=st.session_state.pdf_name,
                top_k=5
            )

        if isinstance(response, dict):  # response returned as dict
            st.subheader("🧠 Agent Response")
            st.markdown(f"**Response:** {response.get('response', '')}")
            st.markdown(f"**Confidence:** {response.get('confidence', 0):.2f}")
        else:
            st.subheader("🧠 Agent Response")
            st.markdown(response)