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

# API Keys and Configurations
OPENAI_API_KEY = "sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh"
OPENAI_API_BASE = "https://api.opentyphoon.ai/v1"
MODEL_NAME = "typhoon-v2-70b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
JSON_PATH = "Jsonfile/M.JSON"
CHAT_HISTORY_FILE = "chat_history_policy.json"
EXCEL_FILE_PATH = r'Data real/Car rate book.xlsx'
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
    """Load car data from Excel file with error handling"""
    try:
        if not os.path.exists(file_path):
            st.warning(f"Car rate book file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_excel(file_path, header=0, dtype=str).fillna('')
        df['MANUYR'] = pd.to_numeric(df['MANUYR'], errors='coerce').astype('Int64')
        df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce').astype('Int64')
        
        try:
            df['FDATEA'] = pd.to_datetime(df['FDATEA'], format='%d-%b-%y', errors='coerce')
            df['LDATEA'] = pd.to_datetime(df['LDATEA'], format='%d-%b-%y', errors='coerce')
        except Exception as e:
            st.warning(f"Warning: Date conversion issue: {e}")
            pass
        
        return df
    except Exception as e:
        st.warning(f"Error loading car data: {e}")
        # Return empty DataFrame as fallback
        return pd.DataFrame()

def format_car_row(row):
    """Format a row of car data for retrieval"""
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

def analyze_question_agent(user_input):
    """
    Enhanced function that analyzes a user question to determine if it relates to Car Rate book or Credit Policy.
    Returns the appropriate data source type and a refined question with simpler error handling.
    """
    try:
        # Use a simple keyword matching approach first to avoid potential API failures
        car_keywords = ["car", "vehicle", "price", "truck", "motorcycle", "brand", "model", 
                       "รถยนต์", "รถเก๋ง", "รถกระบะ", "มอเตอร์ไซค์", "ราคา"]
        
        policy_keywords = ["loan", "credit", "policy", "interest", "requirement", "สินเชื่อ", 
                          "เงินกู้", "ดอกเบี้ย", "หลักประกัน", "นโยบาย", "ctvgmhl"]
        
        car_count = sum(1 for word in car_keywords if word.lower() in user_input.lower())
        policy_count = sum(1 for word in policy_keywords if word.lower() in user_input.lower())
        
        # Simple keyword matching as first check
        if car_count > policy_count:
            data_source = "Car Rate"
            reasoning = f"Keyword matching: {car_count} car keywords vs {policy_count} policy keywords"
            return data_source, user_input, reasoning
        
        # If no strong signal from keywords, try LLM approach with error handling
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=MODEL_NAME,
            temperature=0.7,
            max_tokens=4096,  # Reduced max tokens for more reliable responses
        )
        
        template = """
        You are an AI agent responsible for analyzing user questions and determining whether they should be directed to 
        the Car Rate book search or Credit Policy search. 
        
        User question: {question}
        
        Please output your answer in the format:
        DATA_SOURCE: [Car Rate or Credit Policy]
        REFORMULATED_QUESTION: [Reformulated question if needed, or the original question if clear]
        REASONING: [Brief explanation of your decision]
        LANGUAGE: [Thai or English - determine based on the language of the input question]
        """
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        agent_chain = prompt | llm | StrOutputParser()
        
        result = agent_chain.invoke({"question": user_input})
        lines = result.strip().split('\n')
        
        data_source = None
        reformulated_question = user_input
        reasoning = ""
        
        for line in lines:
            if line.startswith('DATA_SOURCE:'):
                data_source = line.replace('DATA_SOURCE:', '').strip()
            elif line.startswith('REFORMULATED_QUESTION:'):
                reformulated_question = line.replace('REFORMULATED_QUESTION:', '').strip()
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # If LLM didn't return a data source, use the fallback
        if not data_source:
            data_source = "Credit Policy" if policy_count >= car_count else "Car Rate"
        
        return data_source, reformulated_question, reasoning
    except Exception as e:
        # In case of any error, provide a simple fallback
        st.warning(f"Error in question analysis: {e}")
        if any(keyword in user_input.lower() for keyword in car_keywords):
            return "Car Rate", user_input, "Fallback: Basic keyword detection"
        return "Credit Policy", user_input, "Fallback: Default selection"

def get_classification_details(product_group, gcode):
    """Get classification details for a vehicle"""
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
    """Build a detailed car response with classification information"""
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
    """Create embeddings model with error handling"""
    try:
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
    except Exception as e:
        st.warning(f"Error creating embeddings model: {e}")
        return None

@st.cache_resource
def create_car_vector_store():
    """Create car vector store with error handling"""
    try:
        car_data = load_car_data(EXCEL_FILE_PATH)
        if car_data.empty:
            st.warning("Car data is empty. Cannot create vector store.")
            return None, None
            
        texts = [format_car_row(row) for _, row in car_data.iterrows()]
        documents = [Document(page_content=text, metadata={"id": str(i)}) for i, text in enumerate(texts)]
        
        embed_model = create_embeddings_model()
        if embed_model is None:
            st.warning("Embedding model could not be created.")
            return None, None
            
        vector_store = FAISS.from_documents(documents, embed_model)
        
        # Try to save the vector store locally, but continue if it fails
        try:
            vector_store.save_local(VECTOR_STORE_PATH)
        except Exception as save_error:
            st.warning(f"Warning: Could not save vector store: {save_error}")
            
        return vector_store, embed_model
    except Exception as e:
        st.warning(f"Error creating vector store: {e}")
        return None, None

@st.cache_resource
def build_car_rag_chain():
    """Build car RAG chain with robust error handling"""
    try:
        vector_store, _ = create_car_vector_store()
        if vector_store is None:
            return None
            
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=MODEL_NAME,
            temperature=0.7,  # Reduced temperature for more consistent outputs
            max_tokens=4096,  # Reduced max tokens for more reliable responses
        )

        template = """
            You are an AI assistant specialized in car pricing information. Answer questions about car prices based on the provided data.

            Relevant car pricing information:
            {context}

            User question: {question}

            Instructions for answering:
            1. If the question is in Thai, please respond in Thai.
            2. If the question is in English, please respond in English.
            3. Always include PRODUCT GROUP and GCODE if available in your response.
            4. If no relevant information is found, respond with "No relevant information found" in English or "ไม่พบข้อมูลที่เกี่ยวข้อง" in Thai.
            5. Summarize the relevant car pricing information in the same language as the question.

            Answer:
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
    except Exception as e:
        st.warning(f"Error building car RAG chain: {e}")
        return None

def format_value(value):
    """Format a value for display"""
    try:
        if isinstance(value, list):
            return "\n".join([f"- {item}" for item in value]) if value else "No data available"
        elif isinstance(value, dict):
            return "\n".join([f"  {k}: {format_value(v)}" for k, v in value.items()]) if value else "No data available"
        else:
            return str(value or "No data available").replace("\\n", "\n")
    except Exception as e:
        return f"Error formatting value: {e}"

def parse_json_to_docs(data, parent_key="", docs=None):
    """Parse JSON to documents with error handling"""
    try:
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
    except Exception as e:
        st.warning(f"Error parsing JSON to docs: {e}")
        return docs or []

def display_resource_cards():
    """Display resource cards based on detected mode"""
    try:
        if st.session_state.detected_mode == "Car Rate":
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
    except Exception as e:
        st.warning(f"Error displaying resource cards: {e}")

def load_chat_history():
    """Load chat history with error handling"""
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
        st.warning(f"Error loading chat history: {e}")
        try:
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"chats": {}}, f)
        except:
            pass
        return {"chats": {}}

def save_chat_history(history):
    """Save chat history with error handling"""
    try:
        chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
        if chat_dir:
            os.makedirs(chat_dir, exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.warning(f"Error saving chat history: {e}")

def save_chat_to_history(chat_id, role, content):
    """Save chat to history with error handling"""
    try:
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
    except Exception as e:
        st.warning(f"Error saving chat to history: {e}")

def delete_chat_history():
    """Delete chat history with error handling"""
    try:
        chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
        if chat_dir:
            os.makedirs(chat_dir, exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"chats": {}}, f)
        return True
    except Exception as e:
        st.warning(f"Error deleting chat history: {e}")
        return False

def delete_single_chat(chat_id):
    """Delete a single chat with error handling"""
    try:
        history = load_chat_history()
        if "chats" in history and chat_id in history["chats"]:
            del history["chats"][chat_id]
            save_chat_history(history)
            return True
        return False
    except Exception as e:
        st.warning(f"Error deleting chat: {e}")
        return False

@st.cache_resource
def load_llm():
    """Load the LLM with error handling"""
    try:
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=MODEL_NAME,
            temperature=0.7,  # Reduced temperature for more consistent outputs
            max_tokens=4096,  # Reduced max tokens for more reliable responses
        )
    except Exception as e:
        st.warning(f"Error loading LLM: {e}")
        return None

@st.cache_resource
def load_policy_data():
    """Load policy data with robust error handling"""
    try:
        embed_model = create_embeddings_model()
        if embed_model is None:
            st.warning("Embedding model could not be created for policy data.")
            return None
        
        # Check JSON file existence
        if not os.path.exists(JSON_PATH):
            st.warning(f"Policy JSON file not found: {JSON_PATH}")
            return None
        
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            try:
                policy_data = json.load(f)
                documents = parse_json_to_docs(policy_data)
                
                if not documents:
                    st.warning("No documents created from policy data.")
                    return None
                    
                vectorstore = FAISS.from_documents(documents, embed_model)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                prompt_template = """
                    You are an AI assistant specializing in credit policies. Please answer the following question using only the information provided:

                    Relevant Information (Context):     
                    {context}

                    Question:
                    {input}

                    Answer (in English, be concise and specific):
                    """

                llm = load_llm()
                if llm is None:
                    st.warning("LLM could not be loaded for policy data.")
                    return None
                    
                prompt = ChatPromptTemplate.from_template(prompt_template)
                document_chain = create_stuff_documents_chain(llm, prompt)

                return create_retrieval_chain(retriever, document_chain)
            except json.JSONDecodeError as je:
                st.warning(f"Invalid JSON in policy file: {je}")
                return None
    except Exception as e:
        st.warning(f"Error loading policy data: {e}")
        return None

def get_chat_preview(content, max_length=30):
    """Get chat preview with error handling"""
    try:
        if not isinstance(content, str):
            content = str(content)
        words = content.split()
        preview = ' '.join(words[:5])
        return f"{preview[:max_length]}..." if len(preview) > max_length else (preview or "...")
    except Exception as e:
        return "..."

def manage_chat_history():
    """Manage chat history UI with error handling"""
    try:
        with st.sidebar:
            apply_custom_css()
            st.markdown('<h1 style="text-align: center; font-size: 32px;">Chat History</h1>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗪 New Chat", type="primary", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
                    st.session_state.session_vector_store = None
                    st.session_state.detected_mode = None
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                    if delete_chat_history():
                        st.session_state.messages = []
                        st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
                        st.session_state.session_vector_store = None
                        st.session_state.detected_mode = None
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
    except Exception as e:
        st.warning(f"Error managing chat history: {e}")

def normal_response(message_placeholder, text):
    """Display text normally without typewriter effect for more reliability"""
    message_placeholder.markdown(text)

def extract_vehicle_info(response, car_data):
    """Extract vehicle information with error handling"""
    try:
        product_group = ""
        gcode = ""

        # Basic regex patterns for extraction
        pg_patterns = [
            r"PRODUCT GROUP[:\s]+([A-Z])",
            r"กลุ่มผลิตภัณฑ์[:\s]+([A-Z])"
        ]
        
        gcode_patterns = [
            r"GCODE[:\s]+([A-Za-z0-9]+)",
            r"ประเภทรถ[:\s]+([A-Za-z0-9]+)"
        ]
        
        # Try pattern matching
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
        
        # Fallback to keyword matching
        if not product_group:
            response_lower = response.lower()
            if any(keyword in response_lower for keyword in ['motorcycle', 'มอเตอร์ไซค์']):
                product_group = 'M'
                gcode = gcode or 'MC'
            elif any(keyword in response_lower for keyword in ['truck', 'รถบรรทุก']):
                product_group = 'T'
                gcode = gcode or 'T10'
            elif any(keyword in response_lower for keyword in ['car', 'sedan', 'รถยนต์', 'รถเก๋ง']):
                product_group = 'C'
                gcode = gcode or 'CA'
            elif any(keyword in response_lower for keyword in ['pickup', 'รถกระบะ']):
                product_group = 'C'
                gcode = 'P1'

        # Final fallback
        product_group = product_group or 'C'
        gcode = gcode or 'CA'
        
        return product_group, gcode
    except Exception as e:
        st.warning(f"Error extracting vehicle info: {e}")
        return 'C', 'CA'  # Default fallback values

def route_query_to_appropriate_chain(user_input):
    """Route query to appropriate chain with error handling"""
    try:
        data_source, reformulated_question, reasoning = analyze_question_agent(user_input)
        
        # Update session state
        st.session_state.detected_mode = data_source
        
        return reformulated_question, data_source, reasoning
    except Exception as e:
        st.warning(f"Error routing query: {e}")
        return user_input, "Credit Policy", "Error in routing, using default"

def main():
    """Main function with improved error handling and UX"""
    try:
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = f"chat_{int(time.time())}_{os.urandom(4).hex()}"
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = "Auto-detect"
        if "chat_mode_selected" not in st.session_state:
            st.session_state.chat_mode_selected = True
        if "detected_mode" not in st.session_state:
            st.session_state.detected_mode = None
        
        # Load data in background
        with st.spinner("Loading resources..."):
            car_data = load_car_data(EXCEL_FILE_PATH)
            
        # Initialize chat interface
        manage_chat_history()

        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-5be2389/www.sawad.co.th/wp-content/uploads/2020/12/logo.png" width="250">
                <h1 style="font-size: 40px; font-weight: bold; margin-top: 10px;">Srisawad Chatbot</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Chat interface
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Handle user input
        if user_input := st.chat_input("Ask a question about cars or credit policy..."):
            # Save user message
            save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Analysis of question
            with st.spinner("Analyzing your question..."):
                reformulated_question, detected_data_source, reasoning = route_query_to_appropriate_chain(user_input)
            
            # Display AI response
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Show that we're working on it
                    message_placeholder.markdown("Processing your question...")
                    
                    try:
                        # Process based on detected mode
                        if detected_data_source == "Car Rate":
                            # Load car chain on demand to avoid caching issues
                            car_chain = build_car_rag_chain()
                            
                            if not car_chain or car_data.empty:
                                error_msg = "Sorry, I can't access car price information at the moment. Please try again later."
                                normal_response(message_placeholder, error_msg)
                                save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            else: 
                                message_placeholder.markdown("Searching for car price information...")
                                
                                # Process car-related query
                                response = car_chain.invoke(reformulated_question)
                                
                                # A simple check to see if we got a valid response
                                if not response or len(response) < 10:
                                    response = "I couldn't find specific information about this car model or price."
                                
                                product_group, gcode = extract_vehicle_info(response, car_data)
                                full_response = build_car_response(response, product_group, gcode)
                                
                                # Display response and save
                                normal_response(message_placeholder, full_response)
                                save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                                
                                # Show resource cards
                                st.session_state.detected_mode = "Car Rate"
                                display_resource_cards()
                                
                        else:  # Credit Policy
                            # Load policy chain on demand
                            policy_chain = load_policy_data()
                            
                            if not policy_chain:
                                error_msg = "Sorry, I can't access credit policy information at the moment. Please try again later."
                                normal_response(message_placeholder, error_msg)
                                save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            else:
                                message_placeholder.markdown("Searching for policy information...")
                                
                                # Process policy-related query
                                response = policy_chain.invoke({"input": reformulated_question})
                                answer = response.get("answer", "I couldn't find specific information about this policy.")

                                # Collect sources
                                sources = set()
                                for doc in response.get("context", []):
                                    if doc and hasattr(doc, 'metadata'):
                                        source = doc.metadata.get("source")
                                        if source:
                                            sources.add(source)
                                
                                # Format response
                                source_text = "\n\n---\n**Reference Source:**"
                                if sources:
                                    source_text += "\n" + "\n".join(f"- {source}" for source in sources)
                                else:
                                    source_text += "\n- No specific sources found"
                                
                                full_response = answer + source_text
                                
                                # Display response and save
                                normal_response(message_placeholder, full_response)
                                save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                                
                                # Show resource cards
                                st.session_state.detected_mode = "Credit Policy"
                                display_resource_cards()
                                
                    except Exception as e:
                        error_msg = f"I'm sorry, I encountered an error while processing your question. Please try again with a different question."
                        st.error(f"Error details: {str(e)}")
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()