!pip install -q pyngrok langchain_community pypdf streamlit sentence-transformers faiss-cpu

import streamlit as st
import os
import json
import requests
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main():
    st.title("PDF-based Question Answering System")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Save PDF locally
        with open(f"./data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"PDF saved locally as {uploaded_file.name}")

        # Load PDF content
        with st.spinner("Loading PDF content..."):
            text = extract_text_from_pdf(f"./data/{uploaded_file.name}")
        st.write("PDF content:")
        st.write(text)

        # Ask question
        question = st.text_input("Ask your question:")
        if st.button("Ask"):
            answer = get_answer(question, text)
            st.write("Answer:")
            st.write(answer)


def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    return all_splits


def get_answer(question, text):
    llm = initialize_llama3()
    vectorstore = initialize_vectorstore(text)
    chain = initialize_qa_chain(llm, vectorstore)
    response = chain({"query": question})
    return response["result"]


def initialize_llama3():
    llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def initialize_vectorstore(text):
    model_path = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(documents=text, embedding=embeddings)
    return vectorstore


def initialize_qa_chain(llm, vectorstore):
    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.

    {context}

    Question: {question}

    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        verbose=False,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


if __name__ == "__main__":
    main()
