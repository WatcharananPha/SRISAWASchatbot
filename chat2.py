import streamlit as st
import os
import json
import time
import pandas as pd
import re 
import nest_asyncio
from typing import Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

nest_asyncio.apply()

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

class QueryAnalysis(BaseModel):
    data_source: str = Field(
        description="The data source to query: 'Car Rate' for car pricing, 'Credit Policy' for loan/credit information, 'Hybrid' for both, or 'General' for general company info"
    )
    reformulated_question: str = Field(
        description="A reformulated version of the question optimized for the selected data source(s)"
    )
    reasoning: str = Field(
        description="The reasoning behind the choice of data source(s)"
    )
    car_specific_query: str = Field(
        default="",
        description="If hybrid mode, the car-specific part of the question"
    )
    policy_specific_query: str = Field(
        default="",
        description="If hybrid mode, the policy-specific part of the question"
    )
    language: str = Field(
        description="The detected language of the query (e.g., 'Thai' or 'English')"
    )

def analyze_question_agent(user_input: str) -> Tuple[str, str, str, str, str, str]:
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=0.3,
        max_tokens=4096,
    )
    
    parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
    system_template = """
    You are an AI agent specialized in analyzing customer queries about Srisawad Company services.
    
    Your task is to analyze the user's question and determine which data source(s) should be used to answer:
    
    1. "Car Rate" - for questions about vehicle pricing, models, car loans, vehicle types
    2. "Credit Policy" - for questions about loan requirements, interest rates, collateral, policy details
    3. "Hybrid" - for questions that require BOTH car rate AND credit policy information
    4. "General" - for general company information that doesn't fit the above
    
    IMPORTANT:
    - For Hybrid questions, provide separate reformulated questions for each source
    - Pay careful attention to the language: if the question is in Thai, respond in Thai; if in English, respond in English
    - Analyze the question deeply - don't just look for keywords
    
    Examples of each type:
    - Car Rate: "How much is a 2022 Toyota Camry worth?"
    - Credit Policy: "What are the loan requirements for mortgages?"
    - Hybrid: "What's the maximum loan I can get for a Honda City 2021 and what documents do I need?"
    - General: "When was Srisawad Company founded?"
    
    Format your response according to the provided JSON schema.
    """
    
    human_template = """
    Analyze this question: {question}
    
    {format_instructions}
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])
    
    chain = chat_prompt | llm | parser
    result = chain.invoke({
        "question": user_input,
        "format_instructions": parser.get_format_instructions(),
    })
    
    data_source = result.data_source
    reformulated_question = result.reformulated_question
    reasoning = result.reasoning
    car_query = result.car_specific_query
    policy_query = result.policy_specific_query
    language = result.language
    
    return data_source, reformulated_question, reasoning, car_query, policy_query, language

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
        
    embed_model = create_embeddings_model()
    if embed_model is None:
        return None, None
    
    store = InMemoryStore()
    id_key = "doc_id"
    
    texts = [format_car_row(row) for _, row in car_data.iterrows()]
    parent_docs = []
    
    for i, text in enumerate(texts):
        metadata = {
            "id": str(i),
            "type": "car_data"
        }

        row = car_data.iloc[i]
        if 'TYPECOD' in row and not pd.isna(row['TYPECOD']):
            metadata["brand"] = row['TYPECOD']
        if 'MODELCOD' in row and not pd.isna(row['MODELCOD']):
            metadata["model"] = row['MODELCOD']
        if 'MANUYR' in row and not pd.isna(row['MANUYR']):
            metadata["year"] = str(row['MANUYR'])
        if 'GCODE' in row and not pd.isna(row['GCODE']):
            metadata["vehicle_type"] = row['GCODE']
        if 'PRODUCT GROUP' in row and not pd.isna(row['PRODUCT GROUP']):
            metadata["product_group"] = row['PRODUCT GROUP']
        
        doc = Document(page_content=text, metadata=metadata)
        parent_docs.append((str(i), doc))
    
    vectorstore = FAISS.from_documents([doc for _, doc in parent_docs], embed_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=64)
    child_docs = []
    for doc_id, doc in parent_docs:
        child_docs.append((doc_id, doc))
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            child = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "parent_id": doc_id
                }
            )
            child_docs.append((doc_id, child))
        
        metadata = doc.metadata
        car_brand = metadata.get("brand", "")
        car_model = metadata.get("model", "")
        car_year = metadata.get("year", "")
        
        if car_brand and car_model:
            summary = f"Information about {car_brand} {car_model} from year {car_year}"
            child = Document(
                page_content=summary,
                metadata={
                    **doc.metadata,
                    "summary_type": "brand_model",
                    "parent_id": doc_id
                }
            )
            child_docs.append((doc_id, child))
            
    for doc_id, doc in child_docs:
        store.mset([(doc_id, doc)])
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 5},
    )
    
    return retriever, embed_model

@st.cache_resource
def build_car_rag_chain():
    retriever, _ = create_car_vector_store()
    if retriever is None:
        return None
    
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=0.5,
        max_tokens=4096,
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
        6. Be sure to include specific details from the retrieved documents.

        Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs):
        unique_docs = {}
        for doc in docs:
            doc_id = doc.metadata.get("id", "")
            if doc_id not in unique_docs or ("chunk_id" not in doc.metadata and "chunk_id" in unique_docs[doc_id].metadata):
                unique_docs[doc_id] = doc
        
        formatted_texts = []
        for doc_id, doc in unique_docs.items():
            formatted_texts.append(doc.page_content)
            
        return "\n\n".join(formatted_texts)

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
        doc_id = f"policy_{len(docs) if docs else 0}"
        metadata = {"source": parent_key.strip('.'), "doc_id": doc_id}

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
        doc_id = f"policy_{len(docs) if docs else 0}"
        page_content = f"Topic: {parent_key.strip('.')}\n{format_value(data)}"
        metadata = {"source": parent_key.strip('.'), "doc_id": doc_id}
        docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return docs

@st.cache_resource
def load_policy_data():
    embed_model = create_embeddings_model()
    if embed_model is None:
        return (None, None)
        
    if not os.path.exists(JSON_PATH):
        return (None, None)
        
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        policy_data = json.load(f)
    
    parent_documents = parse_json_to_docs(policy_data, docs=[])
    if not parent_documents:
        return (None, None)
    
    store = InMemoryStore()
    id_key = "doc_id"
    parent_docs_with_ids = [(doc.metadata["doc_id"], doc) for doc in parent_documents]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    child_docs = []
    
    for doc_id, doc in parent_docs_with_ids:
        child_docs.append((doc_id, doc))
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            child = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "parent_id": doc_id
                }
            )
            child_docs.append((doc_id, child))
        
        source = doc.metadata.get("source", "")
        if "Topic:" in doc.page_content:
            topic = doc.page_content.split("Topic:", 1)[1].split("\n", 1)[0].strip()
            summary = f"Information about {topic} in credit policy"
            child = Document(
                page_content=summary,
                metadata={
                    **doc.metadata,
                    "summary_type": "topic",
                    "parent_id": doc_id
                }
            )
            child_docs.append((doc_id, child))
    
    for doc_id, doc in child_docs:
        store.mset([(doc_id, doc)])
    
    child_documents = [doc for _, doc in child_docs]
    vectorstore = FAISS.from_documents(child_documents, embed_model)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 5},
    )
    
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=0.5,
        max_tokens=4096,
    )
    
    prompt_template = """
        You are an AI assistant specializing in credit policies. 
        If you are asked in Thai, respond in Thai. If asked in English, respond in English.
        Please answer the following question using only the information provided:

        Relevant Information (Context):     
        {context}

        Question:
        {input}

        Answer (be concise and specific):
    """
    
    def format_docs(docs):
        unique_docs = {}
        for doc in docs:
            doc_id = doc.metadata.get("parent_id", doc.metadata.get("doc_id", ""))
            if doc_id not in unique_docs or ("chunk_id" not in doc.metadata and "chunk_id" in unique_docs[doc_id].metadata):
                unique_docs[doc_id] = doc
                
        formatted_texts = []
        for doc_id, doc in unique_docs.items():
            formatted_texts.append(doc.page_content)
            
        return "\n\n".join(formatted_texts)
        
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def create_hybrid_agent(car_retriever, policy_retriever):
    if not car_retriever or not policy_retriever:
        return None
        
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=0.5,
        max_tokens=4096,
    )
    
    def get_relevant_documents(query_dict):
        car_query = query_dict.get("car_query", "")
        policy_query = query_dict.get("policy_query", "")
        original_query = query_dict.get("original_query", "")
        language = query_dict.get("language", "English")
        
        car_docs = []
        if car_query:
            car_docs = car_retriever.get_relevant_documents(car_query)
            
        policy_docs = []
        if policy_query:
            policy_docs = policy_retriever.get_relevant_documents(policy_query)
        
        car_text = "\n\n".join([doc.page_content for doc in car_docs[:3]])
        policy_text = "\n\n".join([doc.page_content for doc in policy_docs[:3]])
        car_section = f"CAR PRICING INFORMATION:\n{car_text}" if car_text else "No car pricing information found."
        policy_section = f"\nCREDIT POLICY INFORMATION:\n{policy_text}" if policy_text else "No credit policy information found."
        
        combined_text = f"{car_section}\n\n{policy_section}"
        
        return {
            "combined_context": combined_text,
            "original_query": original_query,
            "language": language
        }
    
    template = """
    You are an AI assistant for Srisawad Company that specializes in both car pricing and credit policies.
    
    Answer the user's question by using BOTH the car pricing information AND the credit policy information provided.
    
    Use the following format:
    1. First, provide information about the car (if available)
    2. Next, provide information about the related credit/loan policies (if available)
    3. Finally, combine both pieces of information to give a complete answer
    
    {combined_context}
    
    User question: {original_query}
    
    Important instructions:
    - If the question is in Thai, respond in Thai. If in English, respond in English.
    - Include specific details from both the car information and policy information.
    - Make connections between the car details and the loan policies when possible.
    - If information is missing from either source, clearly state what is missing.
    
    Complete Answer:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["combined_context", "original_query"])
    chain = (
        RunnableLambda(get_relevant_documents) 
        | {
            "combined_context": lambda x: x["combined_context"],
            "original_query": lambda x: x["original_query"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

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
    elif st.session_state.detected_mode == "Credit Policy":
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
    elif st.session_state.detected_mode == "Hybrid":
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
    chat_dir = os.path.dirname(CHAT_HISTORY_FILE)
    if chat_dir:
        os.makedirs(chat_dir, exist_ok=True)
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
        temperature=0.5,
        max_tokens=4096,
    )

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

def normal_response(message_placeholder, text):
    message_placeholder.markdown(text)

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
    data_source, reformulated_question, reasoning, car_query, policy_query, language = analyze_question_agent(user_input)
    st.session_state.detected_mode = data_source
    
    return data_source, reformulated_question, reasoning, car_query, policy_query, language

def main():
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
        unsafe_allow_html=True
    )
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if user_input := st.chat_input("Ask a question about cars or credit policy..."):
        save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        with st.spinner("Analyzing your question..."):
            data_source, reformulated_question, reasoning, car_query, policy_query, language = route_query_to_appropriate_chain(user_input)
        
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Processing your question...")
                
                car_retriever, _ = create_car_vector_store()
                policy_chain, policy_retriever = load_policy_data()
                
                if data_source == "Car Rate":
                    car_chain = build_car_rag_chain()
                    
                    if not car_chain or car_data.empty:
                        error_msg = "Sorry, I can't access car price information at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else: 
                        message_placeholder.markdown("Searching for car price information...")
                        response = car_chain.invoke(reformulated_question)
                        
                        if not response or len(response) < 10:
                            response = "I couldn't find specific information about this car model or price."
                        
                        product_group, gcode = extract_vehicle_info(response, car_data)
                        full_response = build_car_response(response, product_group, gcode)
                        
                        normal_response(message_placeholder, full_response)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        display_resource_cards()
                        
                elif data_source == "Credit Policy":
                    if not policy_chain:
                        error_msg = "Sorry, I can't access credit policy information at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        message_placeholder.markdown("Searching for policy information...")
                        response = policy_chain.invoke(reformulated_question)
                        
                        sources = set()
                        if hasattr(response, 'get') and callable(getattr(response, 'get')):
                            for doc in response.get("context", []):
                                if doc and hasattr(doc, 'metadata'):
                                    source = doc.metadata.get("source")
                                    if source:
                                        sources.add(source)
                            answer = response.get("answer", "I couldn't find specific information about this policy.")
                        else:
                            answer = response

                        source_text = "\n\n---\n**Reference Source:**"
                        if sources:
                            source_text += "\n" + "\n".join(f"- {source}" for source in sources)
                        else:
                            source_text += "\n- No specific sources found"
                        
                        full_response = answer + source_text
                        
                        normal_response(message_placeholder, full_response)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        display_resource_cards()
                
                elif data_source == "Hybrid":
                    message_placeholder.markdown("Processing your question about both car pricing and credit policy...")
                    
                    if car_retriever and policy_retriever:
                        hybrid_agent = create_hybrid_agent(car_retriever, policy_retriever)
                        
                        if hybrid_agent:
                            response = hybrid_agent.invoke({
                                "car_query": car_query or reformulated_question,
                                "policy_query": policy_query or reformulated_question,
                                "original_query": reformulated_question,
                                "language": language
                            })
                            
                            normal_response(message_placeholder, response)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            display_resource_cards()
                        else:
                            error_msg = "Sorry, I couldn't create a hybrid agent to process your question. Please try asking about car rates or credit policies separately."
                            normal_response(message_placeholder, error_msg)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        error_msg = "Sorry, I can't access both car and policy information simultaneously. Please try asking about them separately."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                else: 
                    llm = load_llm()
                    if llm:
                        message_placeholder.markdown("Finding general information...")
                        
                        prompt = f"""
                        You are an AI assistant for Srisawad Company. 
                        Provide helpful information about the company based on your knowledge.
                        
                        User question: {reformulated_question}
                        
                        Answer the question to the best of your ability.
                        If the question is in Thai, respond in Thai. If in English, respond in English.
                        """
                        
                        response = llm.predict(prompt)
                        normal_response(message_placeholder, response)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Sorry, I can't process your question at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()