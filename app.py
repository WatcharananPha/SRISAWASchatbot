import streamlit as st
import os
import nest_asyncio
import time
from llama_parse import LlamaParse
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

nest_asyncio.apply()

st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=1.0,
    max_tokens=8192,
)

parser = LlamaParse(
    api_key="llx-3QORP75OUx11inHUpIy67FLzIgYc0gjfAGKRLDiECXOXkkne",
    result_type="markdown",
    num_workers=1,
    verbose=True,
    language="en",
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="query",
        output_key="result",
        k=5
    )

FAISS_INDEX_PATH = "faiss_index/index.faiss"

def process_uploaded_files(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
        else:
            documents = parser.load_data(file_path)
            for doc in documents:
                all_text += doc.text_resource.text + "\n"
        os.remove(file_path)
    
    return all_text

def get_vector_database(uploaded_files=None):
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    faiss_folder = "faiss_index"
    if os.path.exists(faiss_folder) and not uploaded_files:
        try:
            vector_store = FAISS.load_local(faiss_folder, embed_model, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            return None
    if uploaded_files:
        text_content = process_uploaded_files(uploaded_files)
        if not text_content:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=246
        )
        chunks = text_splitter.split_text(text_content)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(
            documents=documents, 
            embedding=embed_model
        )
        os.makedirs(faiss_folder, exist_ok=True)
        vector_store.save_local(faiss_folder)
        
        return vector_store
        
    return None

def create_chatbot(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={'k': 5}) if vector_db else None
    template = """
    You are an AI assistant specializing in providing information about SriSawad Company. Can add context to response
    Use the following context to answer the question. 
    
    If you don't know the answer, say "I don't know."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    ) if retriever else None

def main():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-99fcfef/www.sawad.co.th/wp-content/uploads/2020/12/logo.png.webp" width="100">
            <h1>Srisawad Chatbot Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_files = st.file_uploader("Upload documents (optional)", accept_multiple_files=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.spinner('Loading knowledge base...'):
        vector_db = get_vector_database(uploaded_files)
        
    if not vector_db and os.path.exists(FAISS_INDEX_PATH):
        st.warning("Failed to load the FAISS index. Please try re-uploading files.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if vector_db:
            chatbot = create_chatbot(vector_db)
            if chatbot:
                response = chatbot({"query": user_input})["result"] 
            else:
                response = "I'm unable to retrieve relevant data, but I'll do my best!"
            st.session_state.memory.save_context(
                {"query": user_input}, 
                {"result": response}
            )
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = response
            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[:i+1])
                time.sleep(0.02)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()