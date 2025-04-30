import streamlit as st
import os
import json
import time
import pandas as pd
import re 
import nest_asyncio

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

nest_asyncio.apply()

os.environ["AZURE_OPENAI_API_KEY"] = 'd8e93330e8384f06aa1c8ace726af49e'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://dataiku-gpt4ommi.openai.azure.com/'

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
JSON_PATH = "Jsonfile/M.JSON"
CHAT_HISTORY_FILE = "chat_history_policy.json"
EXCEL_FILE_PATH = r'Data real/Car rate book.xlsx'
VECTOR_STORE_PATH = "car_rate_vectorstore"
VECTOR_STORE_PATH_POLICY = "carpolicyindex"

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
        return pd.DataFrame()
    df = pd.read_excel(file_path, header=0, dtype=str).fillna('')
    df['MANUYR'] = pd.to_numeric(df['MANUYR'], errors='coerce').astype('Int64')
    df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce').astype('Int64')
    df['FDATEA'] = pd.to_datetime(df['FDATEA'], format='%d-%b-%y', errors='coerce')
    df['LDATEA'] = pd.to_datetime(df['LDATEA'], format='%d-%b-%y', errors='coerce')
    return df

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
                num_value = int(value)
                if num_value > 0:
                    formatted_value = f"{num_value:,}" if col == 'RATE' else str(num_value)
                    parts.append(f"{label}: {formatted_value}")
            else:
                parts.append(f"{label}: {value}")
    return ", ".join(parts) if parts else "Insufficient information"

def analyze_question_agent(user_input):
    car_keywords = ["car", "vehicle", "price", "truck", "motorcycle", "brand", "model", 
                   "รถยนต์", "รถเก๋ง", "รถกระบะ", "มอเตอร์ไซค์", "ราคา"]
    policy_keywords = ["loan", "credit", "policy", "interest", "requirement", "สินเชื่อ", 
                      "เงินกู้", "ดอกเบี้ย", "หลักประกัน", "นโยบาย", "ctvgmhl"]
    car_count = sum(1 for word in car_keywords if word.lower() in user_input.lower())
    policy_count = sum(1 for word in policy_keywords if word.lower() in user_input.lower())
    if car_count > policy_count:
        data_source = "Car Rate"
        reasoning = f"Keyword matching: {car_count} car keywords vs {policy_count} policy keywords"
        return data_source, user_input, reasoning
    llm = AzureChatOpenAI(
        openai_api_version="2024-12-01-preview",
        azure_deployment="dataiku-ssci-gpt-4o", 
        temperature=1.0, 
        max_tokens=4096,  
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
    if not data_source:
        data_source = "Credit Policy" if policy_count >= car_count else "Car Rate"
    if any(keyword in user_input.lower() for keyword in car_keywords) and not data_source:
        return "Car Rate", user_input, "Fallback: Basic keyword detection"
    return data_source, reformulated_question, reasoning

def build_car_response(answer, product_group=None, gcode=None):
    return f"""
{answer}
"""

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

def load_or_create_faiss_index(documents, embed_model, index_path):
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embed_model)
    else:
        vectorstore = FAISS.from_documents(documents, embed_model)
        vectorstore.save_local(index_path)
    return vectorstore

@st.cache_resource
def create_car_vector_store():
    car_data = load_car_data(EXCEL_FILE_PATH)
    texts = [format_car_row(row) for _, row in car_data.iterrows()]
    documents = [Document(page_content=text, metadata={"id": str(i)}) for i, text in enumerate(texts)]
    embed_model = create_embeddings_model()
    vector_store = load_or_create_faiss_index(documents, embed_model, VECTOR_STORE_PATH)
    return vector_store, embed_model

@st.cache_resource
def build_car_rag_chain():
    vector_store, _ = create_car_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = AzureChatOpenAI(
        openai_api_version="2024-12-01-preview",
        azure_deployment="dataiku-ssci-gpt-4o",
        temperature=1.0,
        max_tokens=4096, 
    )
    template = """
    คุณคือ AI ผู้ช่วยเชี่ยวชาญด้านข้อมูลราคารถยนต์ของศรีสวัสดิ์ (Srisawad's car pricing information) หน้าที่ของคุณคือตอบคำถามเกี่ยวกับราคารถยนต์โดยใช้ข้อมูลที่ให้ไว้ใน 'Relevant car pricing information (Context)' เท่านั้น โดยอ้างอิง **โครงสร้างข้อมูลและ Mapping ที่ให้ไว้ด้านล่างนี้** เพื่อทำความเข้าใจข้อมูลใน Context และต้องตอบในรูปแบบที่กำหนดด้านล่างนี้อย่างเคร่งครัด

    **โครงสร้างข้อมูลและ Mapping (Data Schema and Mappings):**

    *   **ความหมายคอลัมน์หลัก (Key Column Meanings):**
        *   `TYPECOD`: ยี่ห้อหลักประกัน (Brand)
        *   `MODELCOD`: รุ่นหลักประกัน (Main Model)
        *   `MODELDESC`: รายละเอียดรุ่นหลักประกัน (Sub Model/Description)
        *   `MANUYR`: ปีผลิต (Manufacturing Year)
        *   `RATE`: ราคาประเมิน (Appraisal Price) - **นี่คือราคาที่ต้องดึงมาแสดง**
        *   `GCODE`: ประเภทย่อยของหลักทรัพย์ (Sub-category Code) - **นี่คือรหัสที่ต้องดึงมาแสดงและหาคำอธิบาย**
        *   `PRODUCT GROUP`: ประเภทหลักของหลักประกัน (Main Category Code) - **นี่คือรหัสที่ต้องดึงมาแสดงและหาคำอธิบาย**
        *   `GEAR`: ระบบขับเคลื่อน (Transmission: Auto, Manual)
        *   `FDATEA`: วันที่เริ่มใช้ราคา
        *   `LDATEA`: วันสุดท้ายที่ใช้ราคา

    *   **Mapping สำหรับ GCODE (ประเภทย่อย):**
        *   `CA`: รถเก๋ง (2-5 ประตู)
        *   `P4`: รถกระบะ (4 ประตู)
        *   `P2`: รถกระบะ (แคป)
        *   `P1`: รถกระบะ (ตอนเดียว)
        *   `MC`: รถมอเตอร์ไซค์
        *   `FT`: รถไถ
        *   `T4`: รถบรรทุก (4 ล้อ)
        *   `T6`: รถบรรทุก (6 ล้อ)
        *   `T10`: รถบรรทุก (10 ล้อ)
        *   `T12`: รถบรรทุก (12 ล้อ)
        *   `VA`: รถตู้
        *   `BS`: รถบัส
        *   `HV`: เครื่องจักรกลหนัก
        *   `OT`: รถอื่นๆ
        *   `N01`: สัญญาหลัก-รอง
        *   `G01`: รถไถโรตารี่
        *   `G03`: เครื่องยนต์เพื่อการเกษตร
        *   `LA`: ที่ดินเปล่า
        *   `LH`: ที่ดินพร้อมสิ่งปลูกสร้าง
        *   `IS`: ประกัน
        *   `P04`: สินเชื่อส่วนบุคคล (กลุ่มบริษัท)
        *   `HR`: รถเกี่ยวข้าว

    *   **Mapping สำหรับ PRODUCT GROUP (ประเภทหลัก):**
        *   `A`: NanoFinance
        *   `P`: PLOAN
        *   `T`: Truck (รถบรรทุก)
        *   `M`: Motocycle (มอเตอร์ไซค์)
        *   `V`: รถเกี่ยวข้าว
        *   `G`: Kubota (รถไถเดินตาม/เกษตร)
        *   `H`: House (บ้าน)
        *   `L`: Land (ที่ดิน)
        *   `I`: Insurance (ประกัน)
        *   `C`: Car (รถยนต์ - เก๋ง, กระบะ, ตู้)

    ---

    **Relevant car pricing information (Context):**
    {context}

    **User question:**
    {question}

    ---

    **คำแนะนำสำหรับการตอบ (Instructions):**

    1.  **ใช้ข้อมูลจาก Context เท่านั้น:** ห้ามใช้ความรู้ภายนอกเด็ดขาด
    2.  **ค้นหาข้อมูลโดยใช้ Schema:** จาก Context ที่ให้มา ให้ค้นหารายละเอียดของรถยนต์ที่ตรงกับคำถามของผู้ใช้ (`{question}`) โดยใช้ **โครงสร้างข้อมูลและ Mapping ด้านบน** ช่วยในการระบุและดึงข้อมูลจากคอลัมน์ต่อไปนี้:
        *   ชื่อรุ่นรถ (พยายามรวม `TYPECOD`, `MODELCOD`, `MODELDESC`, `MANUYR` ถ้ามีใน Context)
        *   ราคาประเมิน (จากคอลัมน์ `RATE`)
        *   **รหัส**ประเภทย่อย (จากคอลัมน์ `GCODE`)
        *   **รหัส**กลุ่มผลิตภัณฑ์หลัก (จากคอลัมน์ `PRODUCT GROUP`)
    3.  **รูปแบบการตอบ (Output Format):** ตอบในรูปแบบนี้ *เท่านั้น*:

        ```
        ราคาของ [ชื่อรุ่นรถที่พบใน Context] ราคา [ราคาประเมิน(RATE)ที่พบใน Context] บาท
        ประเภทย่อยของหลักประกัน (GCODE) : [GCODE ที่พบใน Context] ([คำอธิบาย GCODE จาก Mapping ด้านบน])
        ประเภทหลักของหลักประกัน (PRODUCT GROUP) : [PRODUCT GROUP ที่พบใน Context] ([คำอธิบาย PRODUCT GROUP จาก Mapping ด้านบน])
        ```

        *   แทนที่ `[ชื่อรุ่นรถที่พบใน Context]` และ `[ราคาประเมิน(RATE)ที่พบใน Context]` ด้วยข้อมูลที่หาเจอจริงๆ จาก `{context}`.
        *   สำหรับ `[GCODE ที่พบใน Context]` และ `[PRODUCT GROUP ที่พบใน Context]`: ใส่ **รหัส** ที่ดึงมาจาก `{context}`.
        *   สำหรับ `([คำอธิบาย...จาก Mapping ด้านบน])`: **หลังจาก** ได้รหัส `GCODE` และ `PRODUCT GROUP` จาก `{context}` แล้ว ให้ **ค้นหาคำอธิบายที่ตรงกัน** จากส่วน **"Mapping สำหรับ GCODE"** และ **"Mapping สำหรับ PRODUCT GROUP"** ที่ให้ไว้ *ในพรอมต์นี้* แล้วนำมาใส่ในวงเล็บต่อท้ายรหัส.
        *   **สำคัญ:** หากไม่พบ *รหัส* (`GCODE` หรือ `PRODUCT GROUP`) ใน Context สำหรับรถรุ่นนั้น ให้ใส่คำว่า "ไม่มีข้อมูล" สำหรับทั้งรหัสและคำอธิบาย (เช่น `GCODE : ไม่มีข้อมูล`). หากพบ *รหัส* แต่ *ไม่พบคำอธิบาย* ที่ตรงกันใน Mapping ของพรอมต์นี้ (ซึ่งไม่ควรเกิดขึ้นถ้า Mapping ครบ) ให้ใส่เฉพาะรหัสและวงเล็บว่าง `()` หรือระบุว่า `(ไม่มีคำอธิบายใน Mapping)`.
        *   ตรวจสอบให้แน่ใจว่าดึงค่าตัวเลขราคามาใส่ให้ถูกต้อง และใส่หน่วย "บาท" ต่อท้าย (ถ้าเป็นตัวเลข)

    4.  **ภาษา (Language):**
        *   ถ้าคำถาม `{question}` เป็นภาษาไทย ให้ตอบโดยใช้รูปแบบและข้อความภาษาไทยตามข้อ 3 ทั้งหมด
        *   ถ้าคำถาม `{question}` เป็นภาษาอังกฤษ ให้ปรับรูปแบบเป็นภาษาอังกฤษ และใช้คำอธิบายภาษาอังกฤษจาก Mapping (ถ้ามี) หรือใส่เฉพาะ Code หากไม่มีคำอธิบายภาษาอังกฤษใน Mapping เช่น:
            ```
            Price of [Car Model Found in Context]: [RATE Found in Context] Baht
            Sub-category of collateral (GCODE): [GCODE Found in Context] ([GCODE Description from Mapping])
            Main category of collateral (PRODUCT GROUP): [PRODUCT GROUP Found in Context] ([PRODUCT GROUP Description from Mapping])
            ```
            (หากไม่พบข้อมูล ให้ใช้ "Not specified" หรือ "N/A". หากไม่มี Description ให้ใส่ Code อย่างเดียว หรือ Code กับ `()`)

    5.  **กรณีไม่พบข้อมูลเลย:** หากไม่พบข้อมูลใดๆ ใน `{context}` ที่เกี่ยวข้องกับรถยนต์ที่ถามเลย ให้ตอบเพียงแค่: "ไม่พบข้อมูลที่เกี่ยวข้อง" (สำหรับคำถามภาษาไทย) หรือ "No relevant information found" (สำหรับคำถามภาษาอังกฤษ) **ห้าม**ใช้รูปแบบในข้อ 3

    **Answer:**
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

def display_normal_message(message_placeholder, text):
    message_placeholder.markdown(text)

def clean_assistant_response(message_placeholder, text):
    if "No relevant information found" in text or "ไม่พบข้อมูลที่เกี่ยวข้อง" in text:
        message_placeholder.markdown("I don't have specific information about this topic in my knowledge base.")
        return
    lines = text.split('\n')
    filtered_lines = []
    skip_section = False
    for line in lines:
        if "No specific sources found" in line or "No data available" in line:
            continue
        if "Additional details:" in line and ("Not specified" in "".join(lines[lines.index(line):lines.index(line)+4])):
            skip_section = True
            continue
        if skip_section and line.strip() == "":
            skip_section = False
            continue
        if skip_section:
            continue
        filtered_lines.append(line)
    cleaned_text = "\n".join(filtered_lines)
    if "**Reference Source:**" in cleaned_text and len(cleaned_text.split("**Reference Source:**")[1].strip()) < 5:
        cleaned_text = cleaned_text.split("**Reference Source:**")[0].strip()
    message_placeholder.markdown(cleaned_text)

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

def display_resource_cards():
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

def load_chat_history():
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

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

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
    return AzureChatOpenAI(
        openai_api_version="2024-12-01-preview",
        azure_deployment="dataiku-ssci-gpt-4o",
        temperature=1.0,
        max_tokens=4096,
    )

@st.cache_resource
def load_policy_data():
    embed_model = create_embeddings_model()
    try:
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH_POLICY,
            embed_model,
            allow_dangerous_deserialization=True  # <-- IMPORTANT!
        )
    except Exception as e:
        st.error(f"Error loading policy vector store: {e}")
        # fallback or raise
        raise
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt_template = """
        คุณคือผู้เชี่ยวชาญ AI ด้านนโยบายสินเชื่อของศรีสวัสดิ์ (Srisawad's credit policies) โดยเฉพาะอย่างยิ่งข้อมูลที่ให้ไว้ด้านล่างนี้
        หน้าที่ของคุณคือตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้ไว้ในส่วน 'Relevant Information (Context)' เท่านั้น
        ข้อมูลใน Context ประกอบด้วยรายละเอียดเงื่อนไขสำหรับสินเชื่อประเภทต่างๆ หรือ "เป้า" ที่แตกต่างกัน (เช่น เป้า M, เป้า ก, เป้า F, เป้า H, เป้า L) รวมถึงเงื่อนไขย่อย, เอกสารที่ต้องการ, ข้อห้าม, และอำนาจในการอนุมัติหรือยกเว้น

        หากผู้ใช้ถามเป็นภาษาไทย ให้ตอบเป็นภาษาไทย หากถามเป็นภาษาอังกฤษ ให้ตอบเป็นภาษาอังกฤษ

        Relevant Information (Context):
        {context}

        Question:
        {input}

        คำแนะนำเฉพาะสำหรับการตอบ:
        1.  **จำกัดแหล่งข้อมูล:** ใช้ข้อมูลจาก 'Relevant Information (Context)' ที่ให้มาเท่านั้นในการตอบคำถามทั้งหมด
        2.  **ระบุประเภทสินเชื่อ ("เป้า"):** หากคำถามระบุ "เป้า" ที่เฉพาะเจาะจง (เช่น สินเชื่อเป้า M, เงื่อนไขเป้า ก) ให้ค้นหาคำตอบสำหรับ "เป้า" นั้นๆ หากคำถามไม่ระบุ "เป้า" หรือเป็นคำถามทั่วไป ให้พยายามหาคำตอบที่เป็นกฎเกณฑ์ทั่วไป หรือระบุให้ชัดเจนว่าข้อมูลที่ตอบนั้นมาจาก "เป้า" ใด หรือเงื่อนไขแตกต่างกันอย่างไรระหว่าง "เป้า" ต่างๆ (ถ้าข้อมูลระบุไว้)
        3.  **ค้นหารายละเอียดที่แม่นยำ:** ค้นหาข้อมูลที่ตรงกับคำถามให้มากที่สุด เช่น อัตราดอกเบี้ย, จำนวนงวด, คุณสมบัติผู้กู้/ผู้ค้ำ, เอกสารที่ต้องใช้, อายุรถสูงสุด, เกณฑ์การถือครองหลักประกัน, ข้อจำกัดด้านอาชีพ เป็นต้น
        4.  **ระบุอำนาจยกเว้น:** หากใน Context มีการระบุ "อำนาจยกเว้น" หรือชื่อบุคคล/หน่วยงานที่สามารถอนุมัติยกเว้นเงื่อนไขที่เกี่ยวข้องกับคำถามได้ ให้ระบุข้อมูลส่วนนี้ในคำตอบด้วย
        5.  **กรณีไม่พบข้อมูล:** หากไม่พบคำตอบใน Context ที่ให้มา ให้ตอบอย่างชัดเจนว่า "ข้อมูลนี้ไม่มีอยู่ในรายละเอียดที่ให้มา" หรือ "This information is not found in the provided context"
        6.  **ภาษา:** ตอบคำถามด้วยภาษาเดียวกับที่ผู้ใช้ถาม (ไทย หรือ อังกฤษ)
        7.  **ความชัดเจนและกระชับ:** ตอบคำถามให้ชัดเจน ตรงประเด็น และกระชับที่สุดเท่าที่จะทำได้โดยยังคงความถูกต้องครบถ้วนตาม Context

        Answer:
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
                created_at = pd.to_datetime(created_at_str) if created_at_str else pd.Timestamp.now()
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

def extract_vehicle_info(response, car_data):
    product_group = ""
    gcode = ""
    pg_patterns = [
        r"PRODUCT GROUP[:\s]+([A-Z])",
        r"กลุ่มผลิตภัณฑ์[:\s]+([A-Z])"
    ]
    gcode_patterns = [
        r"GCODE[:\s]+([A-Za-z0-9]+)",
        r"ประเภทรถ[:\s]+([A-Za-z0-9]+)"
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
    product_group = product_group or 'C'
    gcode = gcode or 'CA'
    return product_group, gcode

def route_query_to_appropriate_chain(user_input):
    data_source, reformulated_question, reasoning = analyze_question_agent(user_input)
    st.session_state.detected_mode = data_source
    return reformulated_question, data_source, reasoning

def main():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_chat_id", f"chat_{int(time.time())}_{os.urandom(4).hex()}")
    st.session_state.setdefault("chat_mode", "Auto-detect")
    st.session_state.setdefault("chat_mode_selected", True)
    st.session_state.setdefault("detected_mode", None)
    with st.spinner("Loading resources..."):
        car_data = load_car_data(EXCEL_FILE_PATH)
    manage_chat_history()
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-5be2389/www.sawad.co.th/wp-content/uploads/2020/12/logo.png" width="250">
            <h1 style="font-size: 40px; font-weight: bold; margin-top: 10px;">Srisawad Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    if user_input := st.chat_input("Ask a question about cars or credit policy..."):
        st.session_state.user_question = user_input
        save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        with st.spinner("Analyzing your question..."):
            reformulated_question, detected_data_source, _ = route_query_to_appropriate_chain(user_input)
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Processing your question...")
                if detected_data_source == "Car Rate":
                    car_chain = build_car_rag_chain()
                    if not car_chain or car_data.empty:
                        response = "Sorry, I can't access car price information at the moment. Please try again later."
                    else:
                        response = car_chain.invoke(reformulated_question) or "I couldn't find specific information about this car model or price."
                        if len(response) >= 10:
                            product_group, gcode = extract_vehicle_info(response, car_data)
                            response = build_car_response(response, product_group, gcode)
                    for i in range(len(response)):
                        message_placeholder.markdown(response[:i + 1])
                        time.sleep(0.015)
                    save_chat_to_history(st.session_state.current_chat_id, "assistant", response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.detected_mode = "Car Rate"
                    display_resource_cards()
                else:
                    policy_chain = load_policy_data()
                    if not policy_chain:
                        response = "Sorry, I can't access credit policy information at the moment. Please try again later."
                    else:
                        response = policy_chain.invoke({"input": reformulated_question})
                        answer = response.get("answer", "I couldn't find specific information about this policy.")
                        sources = {doc.metadata.get("source") for doc in response.get("context", []) if hasattr(doc, "metadata")}
                        source_text = "\n\n---\n**Reference Source:**" + ("\n" + "\n".join(f"- {source}" for source in sources) if sources else "\n- No specific sources found")
                        response = answer + source_text
                    for i in range(len(response)):
                        message_placeholder.markdown(response[:i + 1])
                        time.sleep(0.02)
                    save_chat_to_history(st.session_state.current_chat_id, "assistant", response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.detected_mode = "Credit Policy"
                    display_resource_cards()

if __name__ == "__main__":
    main()