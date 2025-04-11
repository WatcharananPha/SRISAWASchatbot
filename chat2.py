import streamlit as st
import os
import json
import time
import pandas as pd
import nest_asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

nest_asyncio.apply()

OPENAI_API_KEY = "sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh"
OPENAI_API_BASE = "https://api.opentyphoon.ai/v1"
MODEL_NAME = "typhoon-v2-70b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
JSON_PATH = "Jsonfile/M.JSON"
RATE_BOOK_PATH = "Data real\Car rate book.xlsx"
CHAT_HISTORY_FILE = "chat_history_policy.json"

st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

def format_value(value):
    if isinstance(value, list):
        return "\n".join([f"- {item}" for item in value]) if value else "ไม่มีข้อมูล"
    elif isinstance(value, dict):
        return "\n".join([f"  {k}: {format_value(v)}" for k, v in value.items()]) if value else "ไม่มีข้อมูล"
    else:
        return str(value or "ไม่มีข้อมูล").replace("\\n", "\n")

def parse_json_to_docs(data, parent_key="", docs=None):
    if docs is None:
        docs = []

    if isinstance(data, dict):
        current_topic = data.get("หัวข้อ", data.get("หัวข้อย่อย", parent_key.strip('.')))
        content_parts = []
        metadata = {"source": parent_key.strip('.')}

        for key, value in data.items():
            current_key = f"{parent_key}{key}" if parent_key else key
            if isinstance(value, (dict, list)) and key not in ["หัวข้อ", "หัวข้อย่อย"]:
                parse_json_to_docs(value, f"{current_key}.", docs)
            elif key not in ["หัวข้อ", "หัวข้อย่อย"]:
                readable_key = key.replace("_", " ").replace("เป้า ", "เป้าหมาย ")
                content_parts.append(f"{readable_key}: {format_value(value)}")

        if content_parts:
            page_content = f"หัวข้อ: {current_topic}\n" + "\n".join(content_parts)
            docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    elif isinstance(data, list) and parent_key:
        page_content = f"หัวข้อ: {parent_key.strip('.')}\n{format_value(data)}"
        metadata = {"source": parent_key.strip('.')}
        docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return docs

def append_static_url_sources(answer: str) -> str:
    static_urls = [
        "https://docs.google.com/spreadsheets/d/10Ol2r3_ZTkSf9KSGCjLjs9J4RepArO3tepwhErKyptI/edit?usp=sharing",
        "https://docs.google.com/spreadsheets/d/1Zxf-8sMZOwo36IWoSPXnyhRN02BFXPVGeLYz5h6t_1s/edit?usp=sharing"
    ]
    url_text = "\n" + "\n".join(f"- {url}" for url in static_urls)
    return f"{answer}\n\n---\n**แหล่งข้อมูลเพิ่มเติม:**{url_text}"

def load_chat_history():
    try:
        chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
        if chat_dir:
            os.makedirs(chat_dir, exist_ok=True)

        if not os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"chats": {}}, f)
            return {"chats": {}}
        
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            return json.loads(content) if content else {"chats": {}}
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        try:
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"chats": {}}, f)
        except:
            pass
        return {"chats": {}}

def save_chat_history(history):
    try:
        chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
        if chat_dir:
            os.makedirs(chat_dir, exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def save_chat_to_history(chat_id, role, content):
    history = load_chat_history()
    if "chats" not in history:
        history["chats"] = {}
    
    if chat_id not in history["chats"]:
        history["chats"][chat_id] = {
            "messages": [],
            "created_at": str(pd.Timestamp.now())
        }
    elif "messages" not in history["chats"][chat_id]:
        history["chats"][chat_id]["messages"] = []
    
    history["chats"][chat_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": str(pd.Timestamp.now())
    })
    save_chat_history(history)

def delete_chat_history():
    chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
    if chat_dir:
        os.makedirs(chat_dir, exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"chats": {}}, f)
    return True

def delete_single_chat(chat_id):
    history = load_chat_history()
    if "chats" in history and chat_id in history["chats"]:
        del history["chats"][chat_id]
        save_chat_history(history)
        return True
    return False

@st.cache_resource
def load_and_process_data(_embeddings):

    if not _embeddings:
        st.error("Embedding model not loaded. Cannot process data.")
        return None
    try:
        all_documents = [] # Initialize to store docs from both sources

        json_dir = os.path.dirname(JSON_PATH)
        os.makedirs(json_dir, exist_ok=True)

        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                policy_data = json.load(f)

            root_key = "นโยบายสินเชื่อ_รวม" if "นโยบายสินเชื่อ_รวม" in policy_data else None
            json_docs = parse_json_to_docs(policy_data.get(root_key, policy_data), parent_key=f"{root_key}." if root_key else "")
            all_documents.extend(json_docs) # Add json docs to combined list

        else:
            st.warning(f"JSON file not found at: {JSON_PATH}. Only processing the rate book.")


        # --- Process Excel File ---
        if os.path.exists(RATE_BOOK_PATH):
            try:
                df_rates = pd.read_excel(RATE_BOOK_PATH)

                # Convert dates to standard format if needed:
                date_cols = ["FDATEA", "LDATEA"]
                for col in date_cols:
                    if col in df_rates.columns:
                        try:
                             df_rates[col] = pd.to_datetime(df_rates[col], format='%d-%b-%y', errors='coerce').dt.strftime('%Y-%m-%d')
                        except Exception as date_err:
                             st.warning(f"Could not convert dates in '{col}' to YYYY-MM-DD format: {date_err}")

                for index, row in df_rates.iterrows():
                    # Create page content by joining non-null values
                    content_parts = [f"{k}: {v}" for k, v in row.items() if pd.notna(v)]
                    page_content = "\n".join(content_parts)

                    metadata = {
                         "source": "Car Rate Book", # Consistent source label
                         "row": index, # Use row as a unique identifier
                         # Include some important fields directly in metadata for later filtering/display
                         "model": row.get("MODELCOD"),
                         "year": row.get("MANUYR"),
                         "rate": row.get("RATE")
                    }
                    doc = Document(page_content=page_content, metadata=metadata)
                    all_documents.append(doc)

            except Exception as e:
                st.error(f"Error processing Excel file ({RATE_BOOK_PATH}): {e}")
                return None
        else:
            st.warning(f"Excel file not found at: {RATE_BOOK_PATH}. Only processing the JSON data.")


        if not all_documents:
            st.error("No documents were successfully created from any data source.")
            return None

        vectorstore = FAISS.from_documents(all_documents, _embeddings)
        st.success(f"Knowledge base loaded. {len(all_documents)} sections processed from JSON and Excel.")
        return vectorstore

    except Exception as e:
        st.error(f"Error processing data/creating vector store: {e}")
        return None

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=1.0,
        max_tokens=8192,
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={},
        query_instruction="Represent this query for retrieving relevant documents: "
    )

@st.cache_resource
def load_and_process_data(_embeddings):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            policy_data = json.load(f)

        if "นโยบายสินเชื่อ_รวม" in policy_data:
            documents = parse_json_to_docs(policy_data["นโยบายสินเชื่อ_รวม"])
        else:
            documents = parse_json_to_docs(policy_data)
        return FAISS.from_documents(documents, _embeddings)

@st.cache_resource
def create_chain(_llm, _retriever):
    prompt_template = """
    You are a helpful AI assistant for Srisawad, specializing in their loan policies and car rate information. Use ONLY the provided context to answer the user's questions in Thai.  If the answer is not directly in the context, state that you cannot find the specific information in the provided policy documents or car rate book.  Do not make up information or use external knowledge. Be concise.

        Context:
        {context}

        Question: {input}

        Answer (Thai):
     """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain = create_retrieval_chain(_retriever, document_chain)
    return retrieval_chain

def apply_custom_css():
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

def get_chat_preview(content, max_length=30):
    if not isinstance(content, str):
        content = str(content)
    words = content.split()
    preview = ' '.join(words[:5])
    return f"{preview[:max_length]}..." if len(preview) > max_length else (preview or "...")

def manage_chat_history():
    with st.sidebar:
        apply_custom_css()
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
        
        chats = history.get("chats", {})

        if not chats:
            st.caption("No past chats.")
        else:
            chat_list = []
            for chat_id, chat_info in chats.items():
                messages = chat_info.get("messages", [])
                created_at_str = chat_info.get("created_at")
                try:
                    created_at = pd.to_datetime(created_at_str) if created_at_str else pd.Timestamp.now()
                except ValueError:
                    created_at = pd.Timestamp.now()

                first_user_message = "..."
                for msg in messages:
                    if msg.get("role") == "user":
                        first_user_message = msg.get("content", "...")
                        break
                chat_list.append((created_at, chat_id, first_user_message))

            chat_list.sort(key=lambda x: x[0], reverse=True)
            chats_by_date = {}
            for created_at, chat_id, first_message in chat_list:
                date_str = created_at.strftime('%Y-%m-%d')
                if date_str not in chats_by_date:
                    chats_by_date[date_str] = []
                chats_by_date[date_str].append((chat_id, first_message))

            for date_str in sorted(chats_by_date.keys(), reverse=True):
                st.markdown(f'<div class="chat-date-header">{date_str}</div>', unsafe_allow_html=True)
                for chat_id, first_message in chats_by_date[date_str]:
                    col_btn, col_del = st.columns([0.85, 0.15])
                    
                    with col_btn:
                        preview_text = get_chat_preview(first_message)
                        if st.button(preview_text, key=f"load_{chat_id}", use_container_width=True):
                            st.session_state.messages = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in chats.get(chat_id, {}).get("messages", [])
                            ]
                            st.session_state.current_chat_id = chat_id
                            st.rerun()
                    
                    with col_del:
                        if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                            if delete_single_chat(chat_id):
                                if st.session_state.get("current_chat_id") == chat_id:
                                    st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
                                    st.session_state.messages = []
                                st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)


def load_excel_as_documents(excel_path: str) -> list:
    documents = []
    df = pd.read_excel(excel_path, sheet_name=None)
    for sheet_name, sheet_data in df.items():
        for index, row in sheet_data.iterrows():
            content_parts = []
            for col, value in row.items():
                if pd.notna(value):
                    content_parts.append(f"{col}: {value}")
            if content_parts:
                content = f"แหล่งที่มา: {sheet_name} (แถวที่ {index + 2})\n" + "\n".join(content_parts)
                doc = Document(page_content=content, metadata={"source": f"{excel_path} [{sheet_name}]"})
                documents.append(doc)
    return documents

def main():
    llm = load_llm()
    embeddings = load_embeddings()
    vectorstore = load_and_process_data(embeddings)
    retriever = None
    retrieval_chain = None
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_chain(llm, retriever)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
    
    manage_chat_history()
    
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-5be2389/www.sawad.co.th/wp-content/uploads/2020/12/logo.png" width="250">
            <h1 style="font-size: 40px; font-weight: bold; margin-top: 10px;">Srisawad Chatbot Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_input := st.chat_input("Ask me anything About SRISAWAD..."):
        save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        if not retrieval_chain:
            error_msg = "ขออภัย ระบบยังไม่พร้อมทำงาน กรุณาตรวจสอบข้อผิดพลาดในการโหลดข้อมูล"
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(error_msg)
            save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("กำลังค้นหาคำตอบ...")
                    
                    response = retrieval_chain.invoke({"input": user_input})
                    answer = response.get("answer", "ขออภัย ไม่พบคำตอบสำหรับคำถามนี้")
                    sources = set()
                    for doc in response.get("context", []):
                        source = doc.metadata.get("source")
                        if source:
                            sources.add(source)
                        
                    source_text = "\n\n---\n**แหล่งข้อมูลที่เกี่ยวข้อง:**"
                    if sources:
                        source_text += "\n" + "\n".join(f"- {source}" for source in sources)
                    else:
                        source_text += "\n- ไม่พบแหล่งข้อมูลที่เฉพาะเจาะจง"
                        
                    full_response = append_static_url_sources(answer + source_text)
                    message_placeholder.markdown(full_response)
                    save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()