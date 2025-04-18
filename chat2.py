import streamlit as st
import os
import json
import time
import pandas as pd
import re 
import nest_asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

nest_asyncio.apply()

OPENAI_API_KEY = "sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh"
OPENAI_API_BASE = "https://api.opentyphoon.ai/v1"
MODEL_NAME = "typhoon-v2-70b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
JSON_PATH = "Jsonfile\M.JSON"
CHAT_HISTORY_FILE = "chat_history_policy.json"
EXCEL_FILE_PATH = r'Data real\Car rate book.xlsx'
VECTOR_STORE_PATH = "car_rate_vectorstore"

CONTNO_TYPE_MAPPING = {
    'T': {'BS': 'BS_Bus', 'FT': 'FT_Tractor', 'HV': 'HV_Heavy Machinery', 
          'OT': 'OT_Other Vehicles', 'T10': 'T10_Truck (10 wheels)', 
          'T12': 'T12_Truck [12 wheels]', 'T6': 'T6_Truck (6 wheels)'},
    'A': {'N01': 'N01_Main-Sub Contract'},
    'C': {'CA': 'CA_Sedan (2-5 doors)', 'P1': 'P1_Pickup Truck (Single Cab)', 
          'P2': 'P2_Pickup Truck (Extended Cab)', 'P4': 'P4_Pickup Truck (4 doors)', 
          'T4': 'T4_Truck (4 wheels)', 'VA': 'VA_Van'},
    'G': {'G01': 'G01_Rotary Tiller', 'G03': 'G03_Agricultural Engine'},
    'H': {'LA': 'LA_Vacant Land', 'LH': 'LH_Land with Buildings'},
    'I': {'IS': 'IS_Insurance'},
    'L': {'LA': 'LA_Vacant Land', 'LH': 'LH_Land with Buildings'},
    'M': {'MC': 'MC_Motorcycle'},
    'P': {'P04': 'P04_PLoan_Personal Loan (Company Group)'},
    'V': {'HR': 'HR_Rice Harvester'}
}

PRODUCT_GROUP_MAPPING = {
    'A': 'NanoFinance', 'P': 'PLOAN', 'T': 'Truck', 
    'M': 'Motorcycle', 'V': 'Rice Harvester',
    'G': 'Kubota Walking Tractor', 'H': 'House', 
    'L': 'Land', 'I': 'Insurance', 'C': 'Car'
}

st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

def apply_custom_css():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 450px !important;
                max-width: 450px !important;
                width: 450px !important;
                transition: all 0.3s ease-in-out;
            }
            [data-testid="stSidebarNav"] { display: none; }
            section[data-testid="stSidebarContent"] {
                width: 450px !important;
                padding-right: 1rem;
            }
            button[data-testid="baseButton-secondary"] { visibility: hidden; }
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
            [data-testid="stSidebar"][aria-expanded="false"] {
                margin-left: -450px;
            }
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
            div.st-cc { padding: 0.25rem; }
            div.stRadio > label {
                background-color: #f0f2f6;
                padding: 10px 15px;
                border-radius: 5px;
                margin-right: 10px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            div.stRadio > label:hover { background-color: #e0e2e6; }
            div.stRadio [data-testid="stMarkdownContainer"] p {
                font-size: 16px;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_car_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Car rate book file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path, header=0, dtype=str).fillna('')
        df['MANUYR'] = pd.to_numeric(df['MANUYR'], errors='coerce').astype('Int64')
        df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce').astype('Int64')
        
        try:
            df['FDATEA'] = pd.to_datetime(df['FDATEA'], format='%d-%b-%y', errors='coerce')
            df['LDATEA'] = pd.to_datetime(df['LDATEA'], format='%d-%b-%y', errors='coerce')
        except:
            pass
        
        return df
    except Exception as e:
        st.error(f"Error loading car data: {e}")
        return pd.DataFrame()

def format_car_row(row):
    columns_labels = {
        'TYPECOD': 'Brand', 'MODELCOD': 'Main Model', 'MODELDESC': 'Sub Model',
        'MANUYR': 'Year', 'GEAR': 'Transmission', 'GCODE': 'Vehicle Type',
        'PRODUCT GROUP': 'Product Group', 'RATE': 'Appraisal Price'
    }

    parts = []
    for col, label in columns_labels.items():
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            if col in ['RATE', 'MANUYR']:
                try:
                    num_value = int(value)
                    if num_value > 0:
                        formatted_value = f"{num_value:,}" if col == 'RATE' else str(num_value)
                        parts.append(f"{label}: {formatted_value}")
                except (ValueError, TypeError):
                     parts.append(f"{label}: {value}")
            else:
                parts.append(f"{label}: {value}")
    return ", ".join(parts) if parts else "Insufficient information"

def get_classification_details(product_group, gcode):
    product_group_desc = PRODUCT_GROUP_MAPPING.get(product_group, 'Not specified')
    gcode_desc = "Not specified"
    contno_type = "Not specified"
    
    for type_key, gcode_dict in CONTNO_TYPE_MAPPING.items():
        if gcode in gcode_dict:
            gcode_desc = gcode_dict[gcode]
            contno_type = type_key
            break
            
    return {
        "CONTNO_TYPE": contno_type,
        "GCODE_Description": gcode_desc,
        "Product_Group_Description": f"{product_group}-{product_group_desc}"
    }

def build_car_response(answer, product_group, gcode):
    classification = get_classification_details(product_group, gcode)
    
    return f"""
{answer}

Additional details:
- Contract type (CONTNO_TYPE): {classification['CONTNO_TYPE']}
- Code and subcategory (GCODE): {classification['GCODE_Description']}
- Product group: {classification['Product_Group_Description']}
"""

@st.cache_resource
def create_embeddings_model():
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="Represent this query for retrieving relevant documents: "
    )

@st.cache_resource
def create_embeddings_model():
    class SafeHuggingFaceBgeEmbeddings(HuggingFaceBgeEmbeddings):
        def embed_query(self, text):
            if text is None:
                text = ""
            return super().embed_query(text)
            
    return SafeHuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="Represent this query for retrieving relevant documents: "
    )

@st.cache_resource
def create_car_vector_store():
    car_data = load_car_data(EXCEL_FILE_PATH)
    if car_data.empty:
        return None, None
        
    texts = [format_car_row(row) for _, row in car_data.iterrows()]
    documents = [Document(page_content=text, metadata={"id": str(i)}) for i, text in enumerate(texts)]
    
    embed_model = create_embeddings_model()
    vector_store = FAISS.from_documents(documents, embed_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store, embed_model

@st.cache_resource
def build_car_rag_chain():
    vector_store, _ = create_car_vector_store()
    if not vector_store:
        return None
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=1.0,
        max_tokens=8192,
    )

    template = """
        You are an AI assistant specialized in car pricing information. Your role is to answer questions about car prices based solely on the provided data.

        Relevant car pricing information:
        {context}

        User question: {question}

        Answer: (Please respond in English by summarizing the 'Relevant car pricing information' that best matches the question.
        Always include the following details:
        1. PRODUCT GROUP (example: PRODUCT GROUP: M for motorcycles, C for cars, T for trucks and tractors)
        2. GCODE (example: GCODE: MC for motorcycles, CA for sedans, FT for tractors, P1/P2/P4 for pickup trucks)

        Do not add information that is not present in the provided data. If no relevant information is found, respond with "No relevant information found.")
        """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def format_value(value):
    if isinstance(value, list):
        return "\n".join([f"- {item}" for item in value]) if value else "No data available"
    elif isinstance(value, dict):
        return "\n".join([f"  {k}: {format_value(v)}" for k, v in value.items()]) if value else "No data available"
    else:
        return str(value or "No data available").replace("\\n", "\n")

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
                readable_key = key.replace("_", " ").replace("เป้า ", "Target ")
                content_parts.append(f"{readable_key}: {format_value(value)}")

        if content_parts:
            page_content = f"Topic: {current_topic}\n" + "\n".join(content_parts)
            docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    elif isinstance(data, list) and parent_key:
        page_content = f"Topic: {parent_key.strip('.')}\n{format_value(data)}"
        metadata = {"source": parent_key.strip('.')}
        docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return docs

def append_static_url_sources(answer: str) -> str:
    return f"{answer}\n\n---\n**Additional resources :**"

def display_resource_cards():
    if st.session_state.chat_mode == "Car Rate":
        st.markdown("""
            <div style="margin-top: 20px;">
                <div style="display: flex; justify-content: center; gap: 20px;">
                    <a href="https://docs.google.com/spreadsheets/d/1Zxf-8sMZOwo36IWoSPXnyhRN02BFXPVGeLYz5h6t_1s/edit?usp=sharing" target="_blank" style="text-decoration: none; color: inherit;">
                        <div style="text-align: center; width: 150px;">
                            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); display: flex; flex-direction: column; align-items: center; height: 150px;">
                                <img src="https://cdn-icons-png.flaticon.com/512/888/888850.png" width="64" height="64">
                                <p style="margin-top: 10px; font-weight: 500; font-size: 14px;">Car rate book</p>
                            </div>
                        </div>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="margin-top: 20px;">
                <div style="display: flex; justify-content: center; gap: 20px;">
                    <a href="https://docs.google.com/spreadsheets/d/10Ol2r3_ZTkSf9KSGCjLjs9J4RepArO3tepwhErKyptI/edit?usp=sharing" target="_blank" style="text-decoration: none; color: inherit;">
                        <div style="text-align: center; width: 150px;">
                            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); display: flex; flex-direction: column; align-items: center; height: 150px;">
                                <img src="https://cdn-icons-png.flaticon.com/512/888/888850.png" width="64" height="64">
                                <p style="margin-top: 10px; font-weight: 500; font-size: 14px;">Credit Policy - CTVGMHL</p>
                            </div>
                        </div>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
def load_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=1.0,
        max_tokens=8192,
    )

@st.cache_resource
def load_policy_data():
    embed_model = create_embeddings_model()
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        policy_data = json.load(f)
        documents = parse_json_to_docs(policy_data)
    
    vectorstore = FAISS.from_documents(documents, embed_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
        You are an AI assistant specializing in credit policies. Please answer the following question using only the information provided, and add relevant context to enhance understanding:

        Relevant Information (Context):     
        {context}

        Question:
        {input}

        Answer (in English):
        """

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, document_chain)

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
                st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
                st.session_state.session_vector_store = None
                st.rerun()
        with col2:
            if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                if delete_chat_history():
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
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

def typewriter_effect(message_placeholder, text):
    for i in range(len(text)):
        message_placeholder.markdown(text[:i+1])
        time.sleep(0.015) 

def extract_vehicle_info(response, car_data):
    product_group = ""
    gcode = ""

    pg_patterns = [
        r"PRODUCT GROUP[:\s]+([A-Z])",
        r"กลุ่มผลิตภัณฑ์[:\s]+([A-Z])",
        r"Product Group[:\s]+([A-Z])"
    ]
    
    gcode_patterns = [
        r"GCODE[:\s]+([A-Za-z0-9]+)",
        r"ประเภทรถ[:\s]+([A-Za-z0-9]+)",
        r"[Gg]code[:\s]+([A-Za-z0-9]+)"
    ]
    
    for pattern in pg_patterns:
        matches = re.search(pattern, response)
        if matches:
            product_group = matches.group(1)
            break
    
    for pattern in gcode_patterns:
        matches = re.search(pattern, response)
        if matches:
            gcode = matches.group(1)
            break
    
    if not product_group or not gcode:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not product_group and ('PRODUCT GROUP' in line or 'กลุ่มผลิตภัณฑ์' in line):
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    product_group = parts[1].strip()[0]
            
            if not gcode and ('GCODE' in line or 'ประเภทรถ' in line):
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    gcode_parts = parts[1].strip().split()
                    gcode = gcode_parts[0] if gcode_parts else ""
    
    if not product_group:
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in ['มอเตอร์ไซค์', 'motorcycle', 'wave', 'click', 'scoopy']):
            product_group = 'M'
            gcode = gcode or 'MC'
        elif any(keyword in response_lower for keyword in ['รถบรรทุก', 'truck', 'รถไถ', 'tractor']):
            product_group = 'T'
            gcode = gcode or 'FT' if 'รถไถ' in response_lower else 'T10'
        elif any(keyword in response_lower for keyword in ['รถเก๋ง', 'รถยนต์', 'car', 'sedan']):
            product_group = 'C'
            gcode = gcode or 'CA'
        elif any(keyword in response_lower for keyword in ['รถกระบะ', 'pickup']):
            product_group = 'C'
            gcode = 'P1' if 'ตอนเดียว' in response_lower else ('P2' if 'แคป' in response_lower else 'P4')

    if not product_group and not car_data.empty and 'PRODUCT GROUP' in car_data.columns:
        first_value = car_data['PRODUCT GROUP'].iloc[0]
        if isinstance(first_value, str) and first_value:
            product_group = first_value[0] if first_value else 'C'
    
    if not gcode and not car_data.empty and 'GCODE' in car_data.columns:
        gcode = car_data['GCODE'].iloc[0] if not car_data.empty else 'CA'

    product_group = product_group or 'C'
    gcode = gcode or 'CA'
    
    return product_group, gcode

def main():
    car_chain = build_car_rag_chain()
    car_data = load_car_data(EXCEL_FILE_PATH)
    policy_chain = load_policy_data()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "Credit Policy"
    if "chat_mode_selected" not in st.session_state:
        st.session_state.chat_mode_selected = False
    
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
    
    if not st.session_state.chat_mode_selected and not st.session_state.messages:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Select chat feature</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer;">
                    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center; margin: 0 auto; border: 2px solid #e9ecef;">
                        <span style="font-size: 40px;">📋</span>
                    </div>
                    <p style="text-align: center; font-weight: bold; margin-top: 10px;">Credit Policy - CTVGMHL</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Credit Policy", key="credit_policy_btn", use_container_width=True):
                st.session_state.chat_mode = "Credit Policy"
                st.session_state.chat_mode_selected = True
                st.rerun()
        
        with col2:
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer;">
                    <div style="width: 100px; height: 100px; border-radius: 50%; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center; margin: 0 auto; border: 2px solid #e9ecef;">
                        <span style="font-size: 40px;">🚗</span>
                    </div>
                    <p style="text-align: center; font-weight: bold; margin-top: 10px;">Car rate book</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Car Rate", key="car_rate_btn", use_container_width=True):
                st.session_state.chat_mode = "Car Rate"
                st.session_state.chat_mode_selected = True
                st.rerun()
    else:
        current_mode_label = "Credit Policy - CTVGMHL" if st.session_state.chat_mode == "Credit Policy" else "Car rate book"
        current_icon = "📋" if st.session_state.chat_mode == "Credit Policy" else "🚗"

        mode_container = st.container()
        with mode_container:
            left_col, right_col = st.columns([0.85, 0.15])
            
            with left_col:
                st.markdown(f"""
                    <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 8px 12px; border-radius: 8px;">
                        <div style="width: 28px; height: 28px; border-radius: 50%; background-color: #e9ecef; 
                        display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                            <span style="font-size: 16px;">{current_icon}</span>
                        </div>
                        <span style="font-weight: bold;">Current mode: {current_mode_label}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            with right_col:
                if st.button("Change", key="change_mode_btn", use_container_width=True):
                    if st.session_state.chat_mode == "Credit Policy":
                        st.session_state.chat_mode = "Car Rate"
                    else:
                        st.session_state.chat_mode = "Credit Policy"
                    
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
                    st.rerun()

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if user_input := st.chat_input(f"Ask a question about {st.session_state.chat_mode}..."):
            save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

        if st.session_state.chat_mode == "Credit Policy" and user_input:
            if not policy_chain:
                error_msg = "Sorry, the system is not ready. Please check for errors in loading the data."
                with chat_container:
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("Searching for an answer...")
                        
                        try:
                            if not user_input.strip():
                                message_placeholder.markdown("Please ask a question about credit policy")
                                return
                                
                            response = policy_chain.invoke({"input": user_input})
                            answer = response.get("answer", "Sorry, no answer found for this question")

                            sources = set()
                            for doc in response.get("context", []):
                                if doc and hasattr(doc, 'metadata'):
                                    source = doc.metadata.get("source")
                                    if source:
                                        sources.add(source)
                            
                            source_text = "\n\n---\n**Reference Source :**"
                            if sources:
                                source_text += "\n" + "\n".join(f"- {source}" for source in sources)
                            else:
                                source_text += "\n- No specific sources found"
                                
                            full_response = answer + source_text
                            typewriter_effect(message_placeholder, full_response)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            display_resource_cards()
                        except Exception as e:
                            error_msg = f"Error occurred while searching for an answer: {str(e)}"
                            message_placeholder.error(error_msg)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        elif st.session_state.chat_mode == "Car Rate":
            if not car_chain or car_data.empty:
                error_msg = "Sorry, the system is not ready. Please check for errors in loading car price data."
                with chat_container:
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            elif user_input: 
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("Searching for car price information...")
                        
                        try:
                            if not user_input.strip():
                                message_placeholder.markdown("Please ask a question about car prices")
                                return
                                
                            response = car_chain.invoke(user_input)
                            product_group, gcode = extract_vehicle_info(response, car_data)
                            full_response = build_car_response(response, product_group, gcode)
                            typewriter_effect(message_placeholder, full_response)

                            save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            display_resource_cards()
                            
                        except Exception as e:
                            error_msg = f"Error occurred while searching for car price information: {str(e)}"
                            message_placeholder.error(error_msg)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()