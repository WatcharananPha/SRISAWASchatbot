import streamlit as st
import os
import nest_asyncio
import time
import pandas as pd
from datetime import datetime

import PyPDF2
import io
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
import json

nest_asyncio.apply()

st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

MAIN_FAISS_FOLDER = "faiss_index"
UPLOADED_FAISS_FOLDER = "uploaded_faiss_index"
TEMP_UPLOAD_DIR = "temp_streamlit_uploads"

os.makedirs(MAIN_FAISS_FOLDER, exist_ok=True)
os.makedirs(UPLOADED_FAISS_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@st.cache_resource  
def load_embedding_models():
    st_model = SentenceTransformer("BAAI/bge-m3")
    lc_embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return st_model, lc_embed_model

st_model, lc_embed_model = load_embedding_models()
image_data = {
    "ขั้นตอน": "https://www.sawad.co.th/wp-content/uploads/2024/10/452800239_896789245826573_6595247655261158306_n-819x1024.jpg",
    "ขอเพิ่มวงเงินออนไลน์": "https://scontent.fbkk12-4.fna.fbcdn.net/v/t39.30808-6/471255214_8877507648984508_946598690423709697_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=127cfc&_nc_ohc=Hwl14FsD6KsQ7kNvgH9t73V&_nc_oc=AdlP-NscLBQr5xC6O9LFXEDWaQeHBLBJGuAO0qxQHY6qFDfXRl1-qgRItmHIn5xP6bmzLSP449Rj69v4_o0Ba3Z4&_nc_zt=23&_nc_ht=scontent.fbkk12-4.fna&_nc_gid=-w2mEHH-dNKmmxiXiJoAAA&oh=00_AYEE-K8b8Q3b-eflPexlGo6xc21lzl8AL54OKNJaHS5_vA&oe=67F13AB5",
    "เอกสาร": "https://scontent.fbkk12-5.fna.fbcdn.net/v/t1.6435-9/49372192_1985324158202926_1335522292599357440_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=rZze6y4lyHwQ7kNvgExWIqY&_nc_oc=AdnR5Cx9QKLZmEB6VJ8vMwWqyhrZL5kqyxu-3S0zmn3XGK8evwrKL0WaCWASxEPkVzINNLD2hXI0LCvDpO9XazjC&_nc_zt=23&_nc_ht=scontent.fbkk12-5.fna&_nc_gid=8hNFyuKaJw90Gdkr7oa06g&oh=00_AYEhL4EmzLra2C01JcDkjDRB4sz4bwWdD1G7Yi2eGrfI8g&oe=680A2CB5",
    "ประกันภัยรถยนต์": "https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/486135644_1074484228057073_8174681586289252031_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=127cfc&_nc_ohc=5Dh2aGLdmMoQ7kNvgHfvWgs&_nc_oc=AdlWrWObPq0uusPLZeKFc4PaAttTPJPAp-Xf7mCbCrC2nClYldVN7MCP82r7E4tvibJ2IHQmJ7cBtKS-GxL2pT2J&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=NLRQU4IaSV8ZqRE4bnl37g&oh=00_AYHqVxlhwnfqdZYK82aAIXlDdE4GZSW7dCTgo8Yraj1h3w&oe=67E8ADAA"
}
stored_texts = list(image_data.keys())
stored_embeddings = st_model.encode(stored_texts)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=1.0,
    max_tokens=4096,
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

def process_uploaded_files(uploaded_files):
    all_text = ""
    temp_dir = TEMP_UPLOAD_DIR
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                for page in pdf_reader.pages:
                    all_text += page.extract_text() + "\n\n"
            
            elif uploaded_file.name.lower().endswith('.txt'):
                text_content = uploaded_file.getvalue().decode('utf-8')
                all_text += text_content + "\n\n"
            
            st.write(f"Processed file: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return all_text.strip()

@st.cache_resource(ttl=3600)
def get_vector_database(_lc_embed_model, uploaded_files_info):
    main_store = None
    uploaded_store = None
    main_index_path = os.path.join(MAIN_FAISS_FOLDER, "index.faiss") 
    main_pkl_path = os.path.join(MAIN_FAISS_FOLDER, "index.pkl")
    if not (os.path.exists(main_index_path) and os.path.exists(main_pkl_path)):
        return None
        
    try:
        main_store = FAISS.load_local(
            MAIN_FAISS_FOLDER,
            _lc_embed_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Could not load main index: {str(e)}")
        return None
    
    if main_store:
        try:
            combined_store = FAISS.from_documents(
                [Document(page_content="init")],
                embedding=_lc_embed_model
            )
            combined_store.merge_from(main_store)
        except Exception as e:
            st.error(f"Error initializing combined store: {str(e)}")
            return main_store 
        
        if uploaded_files_info:
            uploaded_files = st.session_state.get("uploaded_files_obj", None)
            if uploaded_files:
                text_content = process_uploaded_files(uploaded_files)
                if text_content:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=128,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    chunks = text_splitter.split_text(text_content)
                    if chunks:
                        documents = [Document(page_content=chunk) for chunk in chunks]
                        try:
                            uploaded_store = FAISS.from_documents(
                                documents=documents,
                                embedding=_lc_embed_model
                            )
                            uploaded_store.save_local(UPLOADED_FAISS_FOLDER)
                            combined_store.merge_from(uploaded_store)
                                
                        except Exception as e:
                            st.error(f"Error creating vector store: {str(e)}")
                            return main_store
        elif os.path.exists(os.path.join(UPLOADED_FAISS_FOLDER, "index.faiss")):
            try:
                uploaded_store = FAISS.load_local(
                    UPLOADED_FAISS_FOLDER,
                    _lc_embed_model,
                    allow_dangerous_deserialization=True
                )
                combined_store.merge_from(uploaded_store)
            except Exception as e:
                st.warning(f"Could not load uploaded index: {str(e)}")
                return main_store

        return combined_store

    return main_store

@st.cache_resource(ttl=3600)
def get_qa_chain(_vector_db, _llm, _memory):
    template = """
        You are an AI assistant specializing in providing information about SriSawad Company.
        Use the following context retrieved from uploaded documents to answer the question accurately.
        If the input is in Thai, respond in Thai. If the input is in English, respond in English.
        Do not make up information or use external knowledge.

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

def load_chat_history():
    try:
        with open("chat_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"chats": {}}

def save_chat_history(history):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

def save_chat_to_history(chat_id, role, content):
    history = load_chat_history()
    if chat_id not in history["chats"]:
        history["chats"][chat_id] = {
            "messages": [],
            "created_at": str(pd.Timestamp.now())
        }
    
    history["chats"][chat_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": str(pd.Timestamp.now())
    })
    
    save_chat_history(history)
    
def get_chat_preview(content, max_length=30):
    words = content.split()
    preview = ' '.join(words[:5])
    return f"{preview[:max_length]}..." if len(preview) > max_length else preview

def manage_chat_history():
    with st.sidebar:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    min-width: 400px !important;
                    max-width: 400px !important;
                    width: 400px !important;
                    transition: width 0.3s;
                }
                [data-testid="stSidebarNav"] {
                    display: none;
                }
                section[data-testid="stSidebarContent"] {
                    width: 450px !important;
                    padding-right: 1rem;
                }
                button[data-testid="baseButton-secondary"] {
                    visibility: hidden;
                }
                .stButton button {
                    height: 50px;
                    font-size: 16px;
                }
                [data-testid="stMarkdownContainer"] h1 {
                    font-size: 24px;
                    padding: 10px 0;
                    text-align: center;
                }
                .stSubheader {
                    font-size: 18px;
                    padding: 5px 0;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h1 style="text-align: center; font-size: 32px;">Chat History</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗪 New Chat", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_chat_id = f"chat_{int(time.time())}"
                st.rerun()
        with col2:
            if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                if delete_chat_history():
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}"
                    st.rerun()
        
        st.divider()
        history = load_chat_history()
        
        if history["chats"]:
            chat_data = []
            for chat_id, chat_info in history["chats"].items():
                for msg in chat_info["messages"]:
                    chat_data.append({
                        "ChatID": chat_id,
                        "Role": msg["role"],
                        "Content": msg["content"],
                        "Timestamp": pd.to_datetime(msg["timestamp"])
                    })
            
            st.session_state.chat_history_df = pd.DataFrame(chat_data)
            st.session_state.chat_history_df['Date'] = st.session_state.chat_history_df['Timestamp'].dt.date
            dates = sorted(st.session_state.chat_history_df['Date'].unique(), reverse=True)
            
            for date in dates:
                st.subheader(date.strftime('%Y-%m-%d'))
                day_chats = st.session_state.chat_history_df[
                    st.session_state.chat_history_df['Date'] == date
                ]
                
                for chat_id in day_chats['ChatID'].unique():
                    chat_messages = day_chats[day_chats['ChatID'] == chat_id]
                    first_message = chat_messages[
                        chat_messages['Role'] == 'user'
                    ].iloc[0]['Content']
                    
                    if st.button(
                        f"{get_chat_preview(first_message)}",
                        key=f"chat_button_{chat_id}",
                        use_container_width=True
                    ):
                        st.session_state.messages = [
                            {"role": msg["role"].lower(), "content": msg["content"]}
                            for msg in history["chats"][chat_id]["messages"]
                        ]
                        st.session_state.current_chat_id = chat_id
                        st.rerun()

def delete_chat_history():
    try:
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump({"chats": {}}, f)
        return True
    except Exception:
        return False

def main():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-99fcfef/www.sawad.co.th/wp-content/uploads/2020/12/logo.png.webp" width="300">
            <h1 style="font-size: 40px; font-weight: bold; margin-top: 20px;">Srisawad Chatbot Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    manage_chat_history()
    with st.expander("Extension Feature (Optional)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, etc.)",
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files is not None and ("uploaded_files_obj" not in st.session_state or st.session_state.uploaded_files_obj != uploaded_files):
            st.session_state.uploaded_files_obj = uploaded_files

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = f"chat_{int(time.time())}"
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
        save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
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
                for  i in range(len(response_text)):
                    message_placeholder.markdown(response_text[:i+1])
                    time.sleep(0.02)

        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response_content)
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})

if __name__ == "__main__":
    main()  
