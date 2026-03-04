import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

st.header("My first chatbot (Local GPT-2)")

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

with st.sidebar:
    st.title("Your Docs")
    file = st.file_uploader("Upload", type="pdf")

if file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    st.write("Document preview:")
    st.write(text[:500] + "...")  # show first 500 chars

    # Get user question
    user_question = st.text_input("Ask a question about your document")

    if user_question:
        # For GPT-2 local testing, just combine question + PDF snippet
        prompt = f"Based on the following text:\n{text[:1000]}\n\nAnswer this question: {user_question}"
        
        result = generator(prompt, max_length=200, num_return_sequences=1)
        st.write(result[0]["generated_text"])