import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from pinecone import Pinecone
import os
from langchain.prompts import PromptTemplate
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdfFile):
    text = ""
    if isinstance(pdfFile, bytes):
        # Convert bytes to a BytesIO object
        from io import BytesIO
        pdfFile = BytesIO(pdfFile)

    pdf_reader = PdfReader(pdfFile)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    index_name = "pdf-chatbot"

    # Create or connect to the Pinecone index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine'
        )

    index = pc.Index(index_name)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    for i, chunk in enumerate(text_chunks):
        embedding = embeddings_model.embed_query(chunk)

        # Ensure all embeddings are explicitly converted to lists
        flattened_embedding = [float(val) for val in np.asarray(embedding).flatten()]

        # Create the desired format
        vector_dict = {
            'id': f'vec{i + 1}',
            'values': flattened_embedding,
        }

        # Upsert the vector in the desired format into Pinecone
        index.upsert(vectors=[vector_dict])


def get_conversational_chain():
    prompt_template = """
        Answer the question in detail as much as possible from the provided context, make sure to provide all the 
        details, if the answer is not in the provided context just say, "answer is not available in the context", do not
        provide the wrong answer\n\n
        Context:\n{context}?\n
        Question:\n{question}\n

        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, text_chunks):
    index_name = "pdf-chatbot"

    # Create or connect to the Pinecone index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # Embed the user's question
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_embedding = embeddings_model.embed_query(user_question)

    response = index.query(
        vector=[query_embedding],
        top_k=5,
        include_values=True
    )
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    chain = get_conversational_chain()
    if response.get('matches'):
        response_from_chain = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response_from_chain["output_text"])
    else:
        st.warning("No valid matches found in the Pinecone response.")


def main():
    st.set_page_config("Chat With Pdf", layout="wide")
    st.title("Chat with PDF using Gemini")

    # Use st.session_state to persist text_chunks across sessions
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []

    col1, col2 = st.columns([1, 1])  # Create two columns

    with col1:
        st.header("Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"])
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.text_chunks = text_chunks  # Update session state
                get_vector_store(text_chunks)
                st.success("Done!")

    with col2:
        st.header("Ask a Question")
        user_question = st.text_input("Shoot your question")
        if st.button("Ask"):
            with st.spinner("Processing..."):
                user_input(user_question, st.session_state.text_chunks)


if __name__ == '__main__':
    main()
