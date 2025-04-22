import streamlit as st
from rag_qa import generate_rag_answer

st.set_page_config(page_title="Tesla 10-K Q&A", page_icon="🚗")

st.title("📊 Tesla 10-K AI Assistant")

query = st.text_input("Ask a question about Tesla's filings:")

if query:
    with st.spinner("Generating answer..."):
        answer = generate_rag_answer(query)
        st.success("Answer:")
        st.write(answer)
