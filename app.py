import streamlit as st
import os
import nest_asyncio
import time
import pandas as pd
from typing import List

import PyPDF2
import io
from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
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
TEMP_UPLOAD_DIR = "temp_streamlit_uploads"

os.makedirs(MAIN_FAISS_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@st.cache_resource  
def load_embedding_models():
    st_model = SentenceTransformer("BAAI/bge-m3")
    lc_embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return st_model, lc_embed_model

st_model, lc_embed_model = load_embedding_models()
image_data = {
    "ขั้นตอน": "https://www.sawad.co.th/wp-content/uploads/2024/10/452800239_896789245826573_6595247655261158306_n-819x1024.jpg",
    "ขอเพิ่มวงเงินออนไลน์": "https://scontent.fbkk12-4.fna.fbcdn.net/v/t39.30808-6/482015418_1063665659138930_5894050534545491298_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=127cfc&_nc_ohc=8wDEWQ74uA0Q7kNvwF9fFof&_nc_oc=AdkvyI1zwcFJNjR-2iD4og8udVXWpLN_5uesiBveQP2yfDPTH8TkZArrCO46TtOw4-xiAQpNA96GNIuJaEN14Opv&_nc_zt=23&_nc_ht=scontent.fbkk12-4.fna&_nc_gid=2dJs9jvVw2jVHnzs_QkMJw&oh=00_AfHfIPpx7v7wlIKaR7s0h7dsXSEvowj1FXgyI_LrcJT5sA&oe=67FAB6DE",
    "เอกสาร": "https://scontent.fbkk12-5.fna.fbcdn.net/v/t1.6435-9/49372192_1985324158202926_1335522292599357440_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=rZze6y4lyHwQ7kNvgExWIqY&_nc_oc=AdnR5Cx9QKLZmEB6VJ8vMwWqyhrZL5kqyxu-3S0zmn3XGK8evwrKL0WaCWASxEPkVzINNLD2hXI0LCvDpO9XazjC&_nc_zt=23&_nc_ht=scontent.fbkk12-5.fna&_nc_gid=8hNFyuKaJw90Gdkr7oa06g&oh=00_AYEhL4EmzLra2C01JcDkjDRB4sz4bwWdD1G7Yi2eGrfI8g&oe=680A2CB5",
    "ประกันภัยรถยนต์": "https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/486135644_1074484228057073_8174681586289252031_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=127cfc&_nc_ohc=hHPdndjlsRAQ7kNvwEL6KV7&_nc_oc=AdkfYiCDE3TelwSCVgOkAX6PYICzS4BmyQ5LVU_6tz4FxO3Txhf6l6HnB8pRo9Ds9OdIwmujYo5W3Ex8ItiqYqy-&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=DIq9la1liQ-GEBorkQj_8Q&oh=00_AfEKD1YYoBXXJC1inH7wP6L_AL2fkV1AJtVa_D4AS_dIQQ&oe=67FAB22A"
}
stored_texts = list(image_data.keys())
stored_embeddings = st_model.encode(stored_texts)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=0.5,
    max_tokens=8192,
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="query",
        output_key="result",
        k=3
    )

if "session_vector_store" not in st.session_state:
    st.session_state.session_vector_store = None

def find_best_match(user_input, _st_model, _stored_texts, _stored_embeddings, threshold=0.6):
    input_embedding = _st_model.encode([user_input])[0]
    similarities = np.dot(_stored_embeddings, input_embedding) / (np.linalg.norm(_stored_embeddings, axis=1) * np.linalg.norm(input_embedding))
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]

    if best_similarity >= threshold:
        best_match = _stored_texts[best_index]
        return image_data.get(best_match, None)
    return None

def process_and_vectorize_files(uploaded_files, _lc_embed_model):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", "\t"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )

    all_documents = []
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    processed_files_count = 0

    for uploaded_file in uploaded_files:
        file_chunks = []
        file_name = uploaded_file.name
        file_content_bytes = uploaded_file.getvalue()

        try:
            if file_name.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content_bytes))
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        text = text.replace('\r', '\n')
                        text = ' '.join(text.split())
                        chunks = text_splitter.split_text(text)
                        for i, chunk in enumerate(chunks):
                            file_chunks.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": file_name,
                                        "page": page_num + 1,
                                        "chunk": i + 1,
                                        "chunk_size": len(chunk),
                                        "timestamp": str(pd.Timestamp.now())
                                    }
                                )
                            )

            elif file_name.lower().endswith('.txt'):
                encodings_to_try = ['utf-8', 'tis-620', 'latin-1', 'cp874']
                text = None
                for enc in encodings_to_try:
                    try:
                        text = file_content_bytes.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if text is None:
                    raise ValueError(f"Could not decode {file_name} with common encodings.")

                text = text.replace('\r', '\n')
                text = ' '.join(text.split())
                chunks = text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    file_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_name,
                                "chunk": i + 1,
                                "chunk_size": len(chunk),
                                "timestamp": str(pd.Timestamp.now())
                            }
                        )
                    )

            elif file_name.lower().endswith('.xlsx'):
                excel_data = pd.read_excel(io.BytesIO(file_content_bytes), sheet_name=None)
                for sheet_name, df in excel_data.items():

                    sheet_text = df.to_string(index=False)
                    sheet_text = ' '.join(sheet_text.split())
                    chunks = text_splitter.split_text(sheet_text)
                    for i, chunk in enumerate(chunks):
                        file_chunks.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": file_name,
                                    "sheet": sheet_name,
                                    "chunk": i + 1,
                                    "chunk_size": len(chunk),
                                    "timestamp": str(pd.Timestamp.now())
                                }
                            )
                        )

            elif file_name.lower().endswith('.csv'):
                df = None
                encodings_to_try = ['utf-8', 'tis-620', 'latin-1', 'cp874']
                for enc in encodings_to_try:
                     try:
                         df = pd.read_csv(io.BytesIO(file_content_bytes), encoding=enc, on_bad_lines='warn')
                         break 
                     except (UnicodeDecodeError, pd.errors.ParserError):
                         file_content_bytes.seek(0)
                         continue 
                     except Exception as read_err:
                         st.warning(f"Could not read {file_name} with encoding {enc}: {read_err}")
                         file_content_bytes.seek(0)
                         continue


                if df is None:
                    raise ValueError(f"Could not parse CSV {file_name} with common encodings.")

                csv_text = df.to_string(index=False)
                csv_text = ' '.join(csv_text.split())
                chunks = text_splitter.split_text(csv_text)
                for i, chunk in enumerate(chunks):
                    file_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_name,
                                "chunk": i + 1,
                                "chunk_size": len(chunk),
                                "timestamp": str(pd.Timestamp.now())
                            }
                        )
                    )
            else:
                 st.warning(f"⚠️ Skipping unsupported file type: {file_name}")
                 continue 

            if file_chunks:
                 all_documents.extend(file_chunks)
                 st.write(f"✅ Processed {file_name} - Created {len(file_chunks)} chunks")
                 processed_files_count += 1
            else:
                 st.warning(f"⚠️ No content extracted or processed for {file_name}")


        except Exception as e:
            st.error(f"❌ Error processing {file_name}: {str(e)}")
            continue

    if not all_documents:
        st.error("No documents were successfully processed or created.")
        return None

    try:
        session_store = FAISS.from_documents(
            documents=all_documents,
            embedding=_lc_embed_model
        )
        st.success(f"Successfully processed {processed_files_count} file(s). Created vector store with {len(all_documents)} total chunks.")
        return session_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_main_vector_database(_lc_embed_model):
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
        return main_store
    except Exception as e:
        st.error(f"Could not load main index: {str(e)}")
        return None

def get_combined_retriever(main_db, session_db=None):
    if session_db is None:
        return main_db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
    main_retriever = main_db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5} 
    )
    
    session_retriever = session_db.as_retriever(
        search_type="similarity", 
        search_kwargs={'k': 5}
    )
    
    class CombinedRetriever(BaseRetriever):
        def get_relevant_documents(self, query: str) -> List[Document]:
            session_docs = session_retriever.get_relevant_documents(query)
            main_docs = main_retriever.get_relevant_documents(query)
            combined_docs = session_docs + main_docs
            return combined_docs
            
        async def aget_relevant_documents(self, query: str) -> List[Document]:
            raise NotImplementedError("Async retrieval not implemented")
            
    return CombinedRetriever()

def get_qa_chain(retriever, _llm, _memory):
    template = """
        You are an AI assistant specializing in providing information about SriSawad Company.
        Use the following context retrieved from documents to answer the question accurately.
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

def get_reference_info(source_documents: List[Document], max_references: int = 3) -> str:
    if not source_documents:
        return "No reference information available"
    
    references = []
    for i, doc in enumerate(source_documents[:max_references], 1): 
        source = doc.metadata.get('source', 'Unknown source')
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        reference = f"Reference {i}:\n"
        reference += f"- Source: {source}\n"
        reference += f"- Preview: {preview}"
        references.append(reference)
    
    return "\n\n".join(references)

def format_response(response_dict, query):
    answer = response_dict.get('result', "Sorry, I couldn't generate a response.")
    source_documents = response_dict.get('source_documents', [])
    references = get_reference_info(source_documents)
    image_url = find_best_match(query, st_model, stored_texts, stored_embeddings)
    formatted_response = []
    
    if image_url:
        formatted_response.append(f"![Relevant Image]({image_url})")
    
    formatted_response.append(answer)
    formatted_response.append("\n---\n**Source References:**")
    formatted_response.append(references)
    
    return "\n\n".join(formatted_response)

def summarize_chat_content(messages, max_words=150):
    if not messages:
        return "No messages to summarize"
    
    summary_prompt = f"""
    Please summarize this conversation about Srisawad Company in {max_words} words or less.
    If the conversation is in Thai, provide the summary in Thai.
    If in English, provide the summary in English.
    Focus on key points and outcomes.

    Conversation:
    {' '.join([f"{msg['role']}: {msg['content']}" for msg in messages])}

    Summary:
    """
    
    try:
        return llm.predict(summary_prompt)
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

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
        st.markdown("""
            <style>
                    [data-testid="stSidebar"] {
                    min-width: 450px !important;
                    max-width: 450px !important;
                    width: 450px !important;
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
                    
                /* Existing styles... */
                
                /* Add smooth transition for sidebar */
                [data-testid="stSidebar"] {
                    transition: all 0.3s ease-in-out;
                }

                /* Hide sidebar when not shown */
                [data-testid="stSidebar"][aria-expanded="false"] {
                    margin-left: -450px;
                }
                
                /* Style for delete button */
                .stButton.delete-button button {
                    background-color: transparent;
                    color: #ff4b4b;
                    border: none;
                    padding: 0;
                    height: 30px;
                    width: 30px;
                    border-radius: 50%;
                }
                .stButton.delete-button button:hover {
                    background-color: #ffeded;
                }
            </style>
        """, unsafe_allow_html=True)


        st.markdown('<h1 style="text-align: center; font-size: 32px;">Chat History</h1>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗪 New Chat", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_chat_id = f"chat_{int(time.time())}"
                st.session_state.session_vector_store = None
                st.rerun()
        with col2:
            if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                if delete_chat_history():
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}"
                    st.session_state.session_vector_store = None
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
                st.markdown(f'<div class="chat-header">{date.strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)
                day_chats = st.session_state.chat_history_df[
                    st.session_state.chat_history_df['Date'] == date
                ]

                for chat_id in day_chats['ChatID'].unique():
                    chat_messages = day_chats[day_chats['ChatID'] == chat_id]
                    first_message = chat_messages[
                        chat_messages['Role'] == 'user'
                    ].iloc[0]['Content']

                    col1, col2 = st.columns([8, 1])
                    with col1:
                        with st.expander(f"{get_chat_preview(first_message)}", expanded=False):
                            show_summary = st.checkbox("Show Summary", key=f"summary_toggle_{chat_id}")
                            if show_summary:
                                messages = [
                                    {"role": row["Role"], "content": row["Content"]}
                                    for _, row in chat_messages.iterrows()
                                ]
                                summary = summarize_chat_content(messages)
                                st.markdown("**Chat Summary:**")
                                st.markdown(f"_{summary}_")
                                st.divider()

                            if st.button(
                                "Load Full Chat",
                                key=f"chat_button_{chat_id}",
                                use_container_width=True
                            ):
                                st.session_state.messages = [
                                    {"role": msg["role"].lower(), "content": msg["content"]}
                                    for msg in history["chats"][chat_id]["messages"]
                                ]
                                st.session_state.current_chat_id = chat_id
                                st.session_state.session_vector_store = None
                                st.rerun()

                    with col2:
                        if st.button("🗑️", key=f"delete_{chat_id}", help="Delete Chat"):
                            if delete_single_chat(chat_id):
                                st.session_state.messages = []
                                st.session_state.current_chat_id = f"chat_{int(time.time())}"
                                st.session_state.session_vector_store = None
                                st.rerun()

def delete_chat_history():
    try:
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump({"chats": {}}, f)
        return True
    except Exception:
        return False

def delete_single_chat(chat_id):
    try:
        history = load_chat_history()
        if chat_id in history["chats"]:
            del history["chats"][chat_id]
            save_chat_history(history)
            return True
    except Exception:
        return False
    return False

if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False 

def main():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-5be2389/www.sawad.co.th/wp-content/uploads/2020/12/logo.png" width="300">
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

        if uploaded_files:
            if "uploaded_files_obj" not in st.session_state or st.session_state.uploaded_files_obj != uploaded_files:
                st.session_state.uploaded_files_obj = uploaded_files
                with st.spinner("Processing documents and creating vector embeddings..."):
                    session_store = process_and_vectorize_files(uploaded_files, lc_embed_model)
                    if session_store:
                        st.session_state.session_vector_store = session_store

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = f"chat_{int(time.time())}"

    main_vector_db = get_main_vector_database(lc_embed_model)
    qa_chain = None
    if main_vector_db:
        session_vector_db = st.session_state.session_vector_store
        combined_retriever = get_combined_retriever(main_vector_db, session_vector_db)
        qa_chain = get_qa_chain(combined_retriever, llm, st.session_state.memory)

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
        response_text = "I apologize, but I'm not able to process your request at the moment. "
        
        if not main_vector_db:
            response_text += "The main knowledge base is not loaded properly."
        elif not qa_chain:
            response_text += "The question-answering system is not initialized correctly."
        else:
            try:
                raw_response = qa_chain({"query": user_input})
                if raw_response and "result" in raw_response:
                    response_text = format_response(raw_response, user_input)
                else:
                    response_text = "I couldn't generate a proper response for your query. Please try rephrasing your question."
            except Exception as e:
                response_text = f"An error occurred while processing your request: {str(e)}"

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""

            if response_text.startswith("![Relevant Image]"):
                parts = response_text.split("\n\n", 1)
                if len(parts) == 2:
                    image_part, text_part = parts
                    message_placeholder.markdown(image_part)
                    text_placeholder = st.empty()
                    for i in range(len(text_part)):
                        text_placeholder.markdown(text_part[:i+1])
                        time.sleep(0.01)
                else:
                    message_placeholder.markdown(response_text)
                full_response_content = response_text
            else:
                for i in range(len(response_text)):
                    message_placeholder.markdown(response_text[:i+1])
                    time.sleep(0.02)
                full_response_content = response_text

        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response_content)
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})

if __name__ == "__main__":
    main()