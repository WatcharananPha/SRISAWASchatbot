import streamlit as st
from llama_parse import LlamaParse
import nest_asyncio
nest_asyncio.apply()

from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings

parser = LlamaParse(
    api_key="llx-3QORP75OUx11inHUpIy67FLzIgYc0gjfAGKRLDiECXOXkkne",
    result_type="markdown",
    num_workers=1,
    verbose=True,   
    language="en",
)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=0.7,
    max_tokens=1024
)

def process_uploaded_files(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        documents = parser.load_data(file_path)
        for doc in documents:
            all_text += doc.text_resource.text + "\n"
        os.remove(file_path)
    
    return all_text

def get_vector_database(uploaded_files=None):
    faiss_index_path = "faiss_index"
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    if os.path.exists(os.path.join(faiss_index_path, "index.faiss")) and not uploaded_files:
        st.info("Loading existing FAISS index...")
        vector_store = FAISS.load_local(faiss_index_path, embeddings=embed_model, allow_dangerous_deserialization=True)
        st.success("FAISS index loaded successfully!")
    else:
        st.info("Processing new uploaded files and creating FAISS index...")
        with st.spinner("Processing..."):
            text_content = process_uploaded_files(uploaded_files)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
            chunks = text_splitter.split_text(text_content)
            documents = [Document(page_content=chunk) for chunk in chunks]
            vector_store = FAISS.from_documents(documents=documents, embedding=embed_model)
            os.makedirs(faiss_index_path, exist_ok=True)
            vector_store.save_local(faiss_index_path)
            st.success("FAISS index created and saved successfully!")
        st.warning("No documents uploaded and no existing FAISS index found. Please upload documents.")
        return None
    
    return vector_store

def create_chatbot(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    template = """
    You are an AI assistant for the Bank of Thailand's Nano Finance Regulation.
    Use the following context to answer the question.
    If you don't know the answer, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

st.title("Chat with the Nano Finance docs, powered by FAISS 🦙")
st.button("Check out the full tutorial to build this app in our blog post", help="Link to tutorial (placeholder)")
st.write("🌟 **Ask me a question about the Nano Finance Regulation!**")
uploaded_files = st.file_uploader("Choose PDF or DOC/DOCX files", accept_multiple_files=True, type=["pdf", "doc", "docx"], key="file_uploader")
vector_db = get_vector_database(uploaded_files)

if vector_db:
    if "qa_chain" not in st.session_state or uploaded_files:  # Recreate QA chain if new files are uploaded
        qa_chain = create_chatbot(vector_db)
        st.session_state.qa_chain = qa_chain

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "bot":
            st.write(f"**Bot:** {message['content']}")
        elif message["role"] == "user":
            st.write(f"**You:** {message['content']}")

    # User input and response
    col1, col2 = st.columns([9, 1])
    with col1:
        prompt = st.text_input("", key="user_input", placeholder="What?")
    with col2:
        st.write(">")  # Submit arrow

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(f"**You:** {prompt}")

        with st.spinner("Generating response..."):
            try:
                response = st.session_state.qa_chain({"query": prompt})
                bot_response = response['result']
            except Exception as e:
                bot_response = f"Error: {e}"

        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.write(f"**Bot:** {bot_response}")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Note about document processing
st.info("Note: Upload documents to create or update the FAISS index. Once created, the index is reused for immediate responses.")