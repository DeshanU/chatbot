import streamlit as st
from PyPDF2 import PdfReader
##from langchain_text import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.chains import RetrievalQA
from langchain.chains import RetrievalQA

st.header('my first chatboty')

with st.sidebar:
    st.title('your docs')
    file = st.file_uploader('Upload', type='pdf')

# extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    #break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150, #bring 150 charactors from the last chunk
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    st.write(chunks)

    # generating embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=API_KEY_2)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # creating vector store FAISS
    # - embeddings OpenAI
    # - initializing FAISS
    # - store chunks and embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input('Ask a question about your document')

    # do similarity tests
    if user_question:
        match = vector_store.similarity_search(user_question)

        #generate llm
        # llm = OpenAi(
        #     open_ai_api_key = API_KEY_2,
        #     tempature = 0.1,
        #     max_tokens = 1000,
        #     model_name = "gpt-3.5-turbo"
        # )

        # create local text-generation pipeline
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # output results
        # chain -> take the question, get relavent, pass it to the LLM, output

        # Do similarity search
        # Pass documents to chain
        # Chain runs LLM

        # chain = load_qa_chain(llm, chain_type="stuff")
        # response = chain.run(input_documents=match, question=user_question)

        #--------------------------------------------

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

        response = qa_chain.run(user_question)

        st.write(response)