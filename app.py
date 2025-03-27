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
from sentence_transformers import SentenceTransformer
import numpy as np

nest_asyncio.apply()

st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

FAISS_FOLDER = "faiss_index"
TEMP_UPLOAD_DIR = "temp_streamlit_uploads"
os.makedirs(FAISS_FOLDER, exist_ok=True)

@st.cache_resource
def load_embedding_models():
    st_model = SentenceTransformer("BAAI/bge-m3")
    lc_embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return st_model, lc_embed_model

st_model, lc_embed_model = load_embedding_models()
image_data = {
    "ขั้นตอน": "https://www.sawad.co.th/wp-content/uploads/2024/10/452800239_896789245826573_6595247655261158306_n-819x1024.jpg",
    "ขอเพิ่มวงเงินออนไลน์": "https://scontent.fbkk9-2.fna.fbcdn.net/v/t39.30808-6/486686347_1076360731202756_168808874607018632_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=127cfc&_nc_ohc=uJHbtZIQ3GYQ7kNvgHF4yWW&_nc_oc=AdlPyVfh41AJJVvHcdKikzqXRYeZcfKfAd7PBmC9TDVqDaLXMZ6ht6haqCQqphm58hn5mYtcDqqeXRGhjQYP5ORj&_nc_zt=23&_nc_ht=scontent.fbkk9-2.fna&_nc_gid=gJimbS1Yuc9gZTbg3EWpqA&oh=00_AYEUgeZgzkddThejgHiv-SW7TA0cvTKph9ngdIqKAc1cWA&oe=67E8B68B",
    "เอกสาร": "https://scontent.fbkk12-5.fna.fbcdn.net/v/t1.6435-9/49372192_1985324158202926_1335522292599357440_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=rZze6y4lyHwQ7kNvgExWIqY&_nc_oc=AdnR5Cx9QKLZmEB6VJ8vMwWqyhrZL5kqyxu-3S0zmn3XGK8evwrKL0WaCWASxEPkVzINNLD2hXI0LCvDpO9XazjC&_nc_zt=23&_nc_ht=scontent.fbkk12-5.fna&_nc_gid=8hNFyuKaJw90Gdkr7oa06g&oh=00_AYEhL4EmzLra2C01JcDkjDRB4sz4bwWdD1G7Yi2eGrfI8g&oe=680A2CB5",
    "ประกันภัยรถยนต์": "https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/486135644_1074484228057073_8174681586289252031_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=127cfc&_nc_ohc=5Dh2aGLdmMoQ7kNvgHfvWgs&_nc_oc=AdlWrWObPq0uusPLZeKFc4PaAttTPJPAp-Xf7mCbCrC2nClYldVN7MCP82r7E4tvibJ2IHQmJ7cBtKS-GxL2pT2J&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=NLRQU4IaSV8ZqRE4bnl37g&oh=00_AYHqVxlhwnfqdZYK82aAIXlDdE4GZSW7dCTgo8Yraj1h3w&oe=67E8ADAA"
}
stored_texts = list(image_data.keys())
stored_embeddings = st_model.encode(stored_texts)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=1.25,
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
        k=3
    )

def find_best_match(user_input, _st_model, _stored_texts, _stored_embeddings, threshold=0.6):
    input_embedding = _st_model.encode([user_input])[0]
    similarities = np.dot(_stored_embeddings, input_embedding) / (np.linalg.norm(_stored_embeddings, axis=1) * np.linalg.norm(input_embedding))
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]

    if best_similarity >= threshold:
        best_match = _stored_texts[best_index]
        return image_data.get(best_match, None)
    return None

def process_uploaded_files(uploaded_files, _parser):
    all_text = ""
    temp_dir = TEMP_UPLOAD_DIR
    os.makedirs(temp_dir, exist_ok=True)
    temp_files_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
        temp_files_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            continue

        st.write(f"Processing file: {uploaded_file.name}...")
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n\n"
        else:
            documents = _parser.load_data(file_path)
            for doc in documents:
                all_text += doc.text + "\n\n"
        for temp_path in temp_files_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    return all_text.strip()

@st.cache_resource(ttl=3600)
def get_vector_database(_lc_embed_model, uploaded_files_info):
    faiss_index_path = os.path.join(FAISS_FOLDER, "index.faiss")
    faiss_pkl_path = os.path.join(FAISS_FOLDER, "index.pkl")
    os.makedirs(FAISS_FOLDER, exist_ok=True)
    vector_store = None
    if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path) and not uploaded_files_info:
        try:
            vector_store = FAISS.load_local(
                FAISS_FOLDER,
                _lc_embed_model,
                allow_dangerous_deserialization=True
            )
        except Exception:
            try:
                if os.path.exists(faiss_index_path): os.remove(faiss_index_path)
                if os.path.exists(faiss_pkl_path): os.remove(faiss_pkl_path)
            except Exception:
                pass
            vector_store = None

    if uploaded_files_info:
        uploaded_files = st.session_state.get("uploaded_files_obj", None)
        if not uploaded_files:
             return None 
        text_content = process_uploaded_files(uploaded_files, parser)
        if not text_content:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text_content)
        if not chunks:
            return None

        documents = [Document(page_content=chunk) for chunk in chunks]
        try:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=_lc_embed_model
            )
            try:
                vector_store.save_local(FAISS_FOLDER)
            except Exception:
                pass
        except Exception:
            vector_store = None
    return vector_store

@st.cache_resource(ttl=3600)
def get_qa_chain(_vector_db, _llm, _memory):
    template = """
        You are an AI assistant specializing in providing information about SriSawad Company.
        Use the following context retrieved from uploaded documents to answer the question accurately.
        If the input is in Thai, respond in Thai. If the input is in English, respond in English.

        If you don't know the answer based *only* on the provided context, clearly state that the information is not available in the documents \
        (e.g., "ฉันไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในเอกสารที่ให้มา" or "I don't have information about that in the provided documents."). Do not make up information or use external knowledge.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    retriever = _vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3}
        )

    qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=retriever,
            memory=_memory,
            chain_type_kwargs={
                "prompt": PROMPT
            },
            return_source_documents=True
        )
    return qa_chain

def format_response(response_dict, query):
    answer = response_dict.get('result', "Sorry, I couldn't generate a response.")
    image_url = find_best_match(query, st_model, stored_texts, stored_embeddings)
    if image_url:
        return f"![Relevant Image]({image_url})\n\n{answer}"
    else:
        return answer

def main():
    st.markdown(
         """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-99fcfef/www.sawad.co.th/wp-content/uploads/2020/12/logo.png.webp" width="300">
            <h1>Srisawad Chatbot Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Extension Feature (Optional)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, etc.)",
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_files is not None and ("uploaded_files_obj" not in st.session_state or st.session_state.uploaded_files_obj != uploaded_files):
            st.session_state.uploaded_files_obj = uploaded_files

    uploaded_files_info = tuple((f.name, f.size, f.type) for f in st.session_state.get("uploaded_files_obj", [])) if st.session_state.get("uploaded_files_obj") else None
    vector_db = get_vector_database(lc_embed_model, uploaded_files_info)
    qa_chain = None
    if vector_db:
        qa_chain = get_qa_chain(vector_db, llm, st.session_state.memory)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything About SRISAWAD...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response_text = "Chatbot is not ready. Please upload documents and ensure they are processed."
        if qa_chain:
            try:
                raw_response = qa_chain({"query": user_input})
                response_text = format_response(raw_response, user_input)
            except Exception as e:
                response_text = "Sorry, an error occurred."
        elif vector_db:
             response_text = "Error: KB loaded, but chatbot components failed."
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            if response_text.startswith("![Relevant Image]"):
                parts = response_text.split("\n\n", 1)
                image_part, text_part = parts if len(parts) == 2 else (None, response_text)
                if image_part: message_placeholder.markdown(image_part)
                text_placeholder = st.empty() if image_part else message_placeholder
                current_text = text_part if image_part else response_text
                for i in range(len(current_text)):
                    text_placeholder.markdown(current_text[:i+1])
                    time.sleep(0.01)
                full_response_content = response_text
            else:   
                full_response_content = response_text
                for i in range(len(response_text)):
                    message_placeholder.markdown(response_text[:i+1])
                    time.sleep(0.02)

        st.session_state.messages.append({"role": "assistant", "content": full_response_content})


if __name__ == "__main__":
    main()
