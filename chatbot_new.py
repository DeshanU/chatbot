import streamlit as st
from PyPDF2 import PdfReader

# Text splitter (breaks PDF text into chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector database
from langchain_community.vectorstores import FAISS

# Embeddings model
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM pipeline
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Question answering chain
from langchain_community.chains import RetrievalQA


# ----------------------------
# UI
# ----------------------------

st.title("📚 Learning Chatbot")

# Upload PDF
file = st.file_uploader("Upload PDF", type="pdf")


# ----------------------------
# Process PDF
# ----------------------------

if file:

    # Read PDF
    pdf_reader = PdfReader(file)

    text = ""

    # Extract text page by page
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    st.write("✅ PDF text extracted")

    # ----------------------------
    # Split text into chunks
    # ----------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    st.write("✅ Text split into chunks")

    # ----------------------------
    # Create Embeddings
    # ----------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    st.write("✅ Vector database created")


    # ----------------------------
    # Ask Question
    # ----------------------------

    question = st.text_input("Ask a question about your PDF")

    if question:

        # Find similar document chunks
        docs = vector_store.similarity_search(question)

        # Create LLM
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

        # Generate answer
        response = qa_chain.run(question)

        st.subheader("Answer")
        st.write(response)