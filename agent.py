import streamlit as st
import os
import nest_asyncio
import time
import pandas as pd
import io
import re
import json
from typing import List

import PyPDF2
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

nest_asyncio.apply()

OPENAI_API_KEY = "sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh"
OPENAI_API_BASE = "https://api.opentyphoon.ai/v1"
MODEL_NAME = "typhoon-v2-70b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
MAIN_FAISS_FOLDER = "faiss_index"
TEMP_UPLOAD_DIR = "temp_streamlit_uploads"
JSON_PATH = "Jsonfile/M.JSON"
CHAT_HISTORY_FILE = "chat_history.json"
EXCEL_FILE_PATH = r'Data real/Car rate book.xlsx'
VECTOR_STORE_PATH = "car_rate_vectorstore"

os.makedirs(MAIN_FAISS_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(JSON_PATH) if os.path.dirname(JSON_PATH) else '.', exist_ok=True)

# Mapping dictionaries for car classification
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

# Predefined image data from original chat application
image_data = {
    "ขั้นตอน": "https://www.sawad.co.th/wp-content/uploads/2024/10/452800239_896789245826573_6595247655261158306_n-819x1024.jpg",
    "ขอเพิ่มวงเงินออนไลน์": "https://scontent.fbkk12-4.fna.fbcdn.net/v/t39.30808-6/482015418_1063665659138930_5894050534545491298_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=127cfc&_nc_ohc=8wDEWQ74uA0Q7kNvwF9fFof&_nc_oc=AdkvyI1zwcFJNjR-2iD4og8udVXWpLN_5uesiBveQP2yfDPTH8TkZArrCO46TtOw4-xiAQpNA96GNIuJaEN14Opv&_nc_zt=23&_nc_ht=scontent.fbkk12-4.fna&_nc_gid=2dJs9jvVw2jVHnzs_QkMJw&oh=00_AfHfIPpx7v7wlIKaR7s0h7dsXSEvowj1FXgyI_LrcJT5sA&oe=67FAB6DE",
    "เอกสาร": "https://scontent.fbkk12-5.fna.fbcdn.net/v/t1.6435-9/49372192_1985324158202926_1335522292599357440_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=rZze6y4lyHwQ7kNvgExWIqY&_nc_oc=AdnR5Cx9QKLZmEB6VJ8vMwWqyhrZL5kqyxu-3S0zmn3XGK8evwrKL0WaCWASxEPkVzINNLD2hXI0LCvDpO9XazjC&_nc_zt=23&_nc_ht=scontent.fbkk12-5.fna&_nc_gid=8hNFyuKaJw90Gdkr7oa06g&oh=00_AYEhL4EmzLra2C01JcDkjDRB4sz4bwWdD1G7Yi2eGrfI8g&oe=680A2CB5",
    "ประกันภัยรถยนต์": "https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/486135644_1074484228057073_8174681586289252031_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=127cfc&_nc_ohc=hHPdndjlsRAQ7kNvwEL6KV7&_nc_oc=AdkfYiCDE3TelwSCVgOkAX6PYICzS4BmyQ5LVU_6tz4FxO3Txhf6l6HnB8pRo9Ds9OdIwmujYo5W3Ex8ItiqYqy-&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=DIq9la1liQ-GEBorkQj_8Q&oh=00_AfEKD1YYoBXXJC1inH7wP6L_AL2fkV1AJtVa_D4AS_dIQQ&oe=67FAB22A"
}

# Streamlit configuration
st.set_page_config(
    page_title="Srisawad Chat",
    page_icon="https://companieslogo.com/img/orig/SAWAD.BK-18d7b4df.png?t=1720244493",
    layout="centered"
)

# Apply custom CSS
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

# ----- EMBEDDING AND MODEL LOADING FUNCTIONS -----

@st.cache_resource  
def load_embedding_models():
    """Load sentence transformer model for embeddings"""
    try:
        st_model = SentenceTransformer("BAAI/bge-m3")
        lc_embed_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="Represent this query for retrieving relevant documents: "
        )
        return st_model, lc_embed_model
    except Exception as e:
        st.warning(f"Error loading embedding models: {e}")
        return None, None

@st.cache_resource
def load_llm():
    """Load LLM model with error handling"""
    try:
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            model_name=MODEL_NAME,
            temperature=0.5,
            max_tokens=2048,
        )
    except Exception as e:
        st.warning(f"Error loading LLM: {e}")
        return None

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
        return pd.DataFrame()

# Initialize models
st_model, lc_embed_model = load_embedding_models()
llm = load_llm()
stored_texts = list(image_data.keys())
stored_embeddings = st_model.encode(stored_texts) if st_model else None

# ----- CONVERSATION MEMORY -----

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

# ----- VECTOR STORE AND RETRIEVAL FUNCTIONS -----

def find_best_match(user_input, _st_model, _stored_texts, _stored_embeddings, threshold=0.6):
    """Find best matching image for user query"""
    try:
        if _st_model is None or _stored_texts is None or _stored_embeddings is None:
            return None
            
        input_embedding = _st_model.encode([user_input])[0]
        similarities = np.dot(_stored_embeddings, input_embedding) / (np.linalg.norm(_stored_embeddings, axis=1) * np.linalg.norm(input_embedding))
        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]

        if best_similarity >= threshold:
            best_match = _stored_texts[best_index]
            return image_data.get(best_match, None)
        return None
    except Exception as e:
        st.warning(f"Error finding best match: {e}")
        return None

def process_and_vectorize_files(uploaded_files, _lc_embed_model):
    """Process and vectorize uploaded files"""
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
                         file_content_bytes = io.BytesIO(uploaded_file.getvalue())
                         continue 
                     except Exception as read_err:
                         st.warning(f"Could not read {file_name} with encoding {enc}: {read_err}")
                         file_content_bytes = io.BytesIO(uploaded_file.getvalue())
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
    main_index_path = os.path.join(MAIN_FAISS_FOLDER, "faiss_index/index.faiss") 
    main_pkl_path = os.path.join(MAIN_FAISS_FOLDER, "faiss_index/index.pkl")
    
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
    """Combine retrievers from multiple sources"""
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
    """Create QA chain for Srisawad general information"""
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

# ----- CAR DATA FUNCTIONS -----

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
def create_car_vector_store():
    """Create car vector store with error handling"""
    try:
        car_data = load_car_data(EXCEL_FILE_PATH)
        if car_data.empty:
            st.warning("Car data is empty. Cannot create vector store.")
            return None, None
            
        texts = [format_car_row(row) for _, row in car_data.iterrows()]
        documents = [Document(page_content=text, metadata={"id": str(i)}) for i, text in enumerate(texts)]
        
        if lc_embed_model is None:
            st.warning("Embedding model not available.")
            return None, None
            
        vector_store = FAISS.from_documents(documents, lc_embed_model)
        
        # Try to save the vector store locally, but continue if it fails
        try:
            vector_store.save_local(VECTOR_STORE_PATH)
        except Exception as save_error:
            st.warning(f"Warning: Could not save vector store: {save_error}")
            
        return vector_store, lc_embed_model
    except Exception as e:
        st.warning(f"Error creating vector store: {e}")
        return None, None

@st.cache_resource
def build_car_rag_chain():
    """Build RAG chain for car data with error handling"""
    try:
        vector_store, _ = create_car_vector_store()
        if vector_store is None:
            return None
            
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        if llm is None:
            st.warning("LLM not available.")
            return None

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

# ----- POLICY DATA FUNCTIONS -----

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

@st.cache_resource
def load_policy_data():
    """Load policy data with robust error handling"""
    try:
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
                    
                vectorstore = FAISS.from_documents(documents, lc_embed_model)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                prompt_template = """
                    You are an AI assistant specializing in credit policies. Please answer the following question using only the information provided:

                    Relevant Information (Context):     
                    {context}

                    Question:
                    {input}

                    Answer (in the same language as the question):
                    """

                if llm is None:
                    st.warning("LLM not available for policy data.")
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

# ----- QUESTION ANALYSIS FUNCTION -----

def analyze_question_agent(user_input):
    """
    Enhanced function that analyzes a user question to determine if it relates to Car Rate book or Credit Policy.
    Returns the appropriate data source type and a refined question with simpler error handling.
    """
    try:
        car_keywords = ["car", "vehicle", "price", "truck", "motorcycle", "brand", "model", 
                       "รถยนต์", "รถเก๋ง", "รถกระบะ", "มอเตอร์ไซค์", "ราคา"]
        
        policy_keywords = ["loan", "credit", "policy", "interest", "requirement", "สินเชื่อ", 
                          "เงินกู้", "ดอกเบี้ย", "หลักประกัน", "นโยบาย", "ctvgmhl"]
        
        general_keywords = ["about", "company", "เกี่ยวกับ", "บริษัท", "ศรีสวัสดิ์", "srisawad"]
        
        car_count = sum(1 for word in car_keywords if word.lower() in user_input.lower())
        policy_count = sum(1 for word in policy_keywords if word.lower() in user_input.lower())
        general_count = sum(1 for word in general_keywords if word.lower() in user_input.lower())
        
        # Simple keyword matching as first check
        if car_count > policy_count and car_count > general_count:
            data_source = "Car Rate"
            reasoning = f"Keyword matching: {car_count} car keywords vs {policy_count} policy keywords vs {general_count} general keywords"
            return data_source, user_input, reasoning
        elif policy_count > car_count and policy_count > general_count:
            data_source = "Credit Policy"
            reasoning = f"Keyword matching: {policy_count} policy keywords vs {car_count} car keywords vs {general_count} general keywords"
            return data_source, user_input, reasoning
        elif general_count > car_count and general_count > policy_count:
            data_source = "General"
            reasoning = f"Keyword matching: {general_count} general keywords vs {car_count} car keywords vs {policy_count} policy keywords"
            return data_source, user_input, reasoning
        
        # If no strong signal from keywords, try LLM approach with error handling
        if llm is None:
            # Fallback if LLM not available
            if car_count >= policy_count:
                return "Car Rate", user_input, "Fallback matching: car keywords"
            else:
                return "Credit Policy", user_input, "Fallback matching: policy keywords"
        
        template = """
        You are an AI agent responsible for analyzing user questions and determining whether they should be directed to 
        the Car Rate book search, Credit Policy search, or General company information. 
        
        User question: {question}
        
        Please output your answer in the format:
        DATA_SOURCE: [Car Rate or Credit Policy or General]
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
            max_count = max(car_count, policy_count, general_count)
            if car_count == max_count:
                data_source = "Car Rate"
            elif policy_count == max_count:
                data_source = "Credit Policy"
            else:
                data_source = "General"
        
        return data_source, reformulated_question, reasoning
    except Exception as e:
        # In case of any error, provide a simple fallback
        st.warning(f"Error in question analysis: {e}")
        if any(keyword in user_input.lower() for keyword in car_keywords):
            return "Car Rate", user_input, "Fallback: Basic keyword detection"
        elif any(keyword in user_input.lower() for keyword in general_keywords):
            return "General", user_input, "Fallback: Basic keyword detection"
        return "Credit Policy", user_input, "Fallback: Default selection"

def extract_vehicle_info(response, car_data):
    """Extract vehicle information from response"""
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

# ----- RESPONSE FORMATTING AND DISPLAY -----

def get_reference_info(source_documents: List[Document], max_references: int = 3) -> str:
    """Format reference information from source documents"""
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
    """Format response with references and image if available"""
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

def display_resource_cards(detected_mode):
    """Display resource cards based on detected mode"""
    try:
        if detected_mode == "Car Rate":
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
        elif detected_mode == "Credit Policy":
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

def normal_response(message_placeholder, text):
    """Display text normally without typewriter effect for more reliability"""
    message_placeholder.markdown(text)

def typewriter_response(message_placeholder, text):
    """Display response with typewriter effect"""
    full_response = ""
    
    # Check if we have an image at the top
    if text.startswith("![Relevant Image]"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            image_part, text_part = parts
            message_placeholder.markdown(image_part)
            text_placeholder = st.empty()
            for i in range(len(text_part)):
                text_placeholder.markdown(text_part[:i+1])
                time.sleep(0.01)
            return text
        
    # No image, just do a regular typewriter effect
    for i in range(len(text)):
        full_response = text[:i+1]
        message_placeholder.markdown(full_response)
        time.sleep(0.01)
    
    return full_response

# ----- CHAT HISTORY MANAGEMENT -----

def summarize_chat_content(messages, max_words=150):
    """Summarize chat content"""
    if not messages or llm is None:
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
    """Load chat history with error handling"""
    try:
        if not os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"chats": {}}, f)
            return {"chats": {}}
        
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"chats": {}}

def save_chat_history(history):
    """Save chat history"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.warning(f"Error saving chat history: {e}")

def save_chat_to_history(chat_id, role, content):
    """Save chat message to history"""
    try:
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
    except Exception as e:
        st.warning(f"Error saving chat to history: {e}")
    
def get_chat_preview(content, max_length=30):
    """Get preview of chat content"""
    try:
        words = content.split()
        preview = ' '.join(words[:5])
        return f"{preview[:max_length]}..." if len(preview) > max_length else preview
    except Exception as e:
        return "..."

def delete_chat_history():
    """Delete all chat history"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"chats": {}}, f)
        return True
    except Exception:
        return False

def delete_single_chat(chat_id):
    """Delete a single chat by ID"""
    try:
        history = load_chat_history()
        if chat_id in history["chats"]:
            del history["chats"][chat_id]
            save_chat_history(history)
            return True
    except Exception:
        pass
    return False

def manage_chat_history():
    """Manage chat history sidebar"""
    with st.sidebar:
        apply_custom_css()
        st.markdown('<h1 style="text-align: center; font-size: 32px;">Chat History</h1>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗪 New Chat", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_chat_id = f"chat_{int(time.time())}"
                st.session_state.session_vector_store = None
                st.session_state.detected_mode = None
                st.rerun()
        with col2:
            if st.button("🗑️ Delete All", type="secondary", use_container_width=True):
                if delete_chat_history():
                    st.session_state.messages = []
                    st.session_state.current_chat_id = f"chat_{int(time.time())}"
                    st.session_state.session_vector_store = None
                    st.session_state.detected_mode = None
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

            if chat_data:
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
                        first_message = "..."
                        for _, row in chat_messages.iterrows():
                            if row['Role'] == 'user':
                                first_message = row['Content']
                                break

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
                                    if st.session_state.current_chat_id == chat_id:
                                        st.session_state.messages = []
                                        st.session_state.current_chat_id = f"chat_{int(time.time())}"
                                        st.session_state.session_vector_store = None
                                    st.rerun()

# ----- MAIN APPLICATION -----

def main():
    """Main application function"""
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-cncpm.nitrocdn.com/DpTaQVKLCVHUePohOhFgtgFLWoUOmaMZ/assets/images/optimized/rev-5be2389/www.sawad.co.th/wp-content/uploads/2020/12/logo.png" width="300">
            <h1 style="font-size: 40px; font-weight: bold; margin-top: 20px;">Srisawad Chatbot Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Handle chat history in the sidebar
    manage_chat_history()
    
    # Optional file upload functionality
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

    # Initialize session state for chat
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = f"chat_{int(time.time())}"
    if "detected_mode" not in st.session_state:
        st.session_state.detected_mode = None
    
    # Load main vector database
    main_vector_db = get_main_vector_database(lc_embed_model)
    qa_chain = None
    if main_vector_db:
        session_vector_db = st.session_state.session_vector_store
        combined_retriever = get_combined_retriever(main_vector_db, session_vector_db)
        qa_chain = get_qa_chain(combined_retriever, llm, st.session_state.memory)

    # Initialize messages if not in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    user_input = st.chat_input("Ask me anything about SRISAWAD...")
    if user_input:
        # Save user message to history and display
        save_chat_to_history(st.session_state.current_chat_id, "user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Determine the appropriate query handler using analyze_question_agent
        with st.spinner("Analyzing your question..."):
            data_source, reformulated_question, reasoning = analyze_question_agent(user_input)
        
        # Update detected mode in session state
        st.session_state.detected_mode = data_source

        # Process response based on detected data source
        response_text = "I apologize, but I'm not able to process your request at the moment. "
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Processing your question...")
            
            try:
                if data_source == "Car Rate":
                    # Handle car rate queries
                    car_chain = build_car_rag_chain()
                    
                    if not car_chain:
                        error_msg = "Sorry, I can't access car price information at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else: 
                        # Process car-related query
                        response = car_chain.invoke(reformulated_question)
                        
                        if not response or len(response) < 10:
                            response = "I couldn't find specific information about this car model or price."
                        
                        car_data = load_car_data(EXCEL_FILE_PATH)
                        product_group, gcode = extract_vehicle_info(response, car_data)
                        full_response = build_car_response(response, product_group, gcode)
                        
                        # Display response with typewriter effect
                        typewriter_response(message_placeholder, full_response)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Display resource cards
                        display_resource_cards("Car Rate")
                
                elif data_source == "Credit Policy":
                    # Handle credit policy queries
                    policy_chain = load_policy_data()
                    
                    if not policy_chain:
                        error_msg = "Sorry, I can't access credit policy information at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
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
                        
                        # Display response with typewriter effect
                        typewriter_response(message_placeholder, full_response)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Display resource cards
                        display_resource_cards("Credit Policy")
                
                else:  # General information
                    # Handle general company information queries
                    if not main_vector_db or not qa_chain:
                        error_msg = "Sorry, I can't access general company information at the moment. Please try again later."
                        normal_response(message_placeholder, error_msg)
                        save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        try:
                            raw_response = qa_chain({"query": user_input})
                            if raw_response and "result" in raw_response:
                                response_text = format_response(raw_response, user_input)
                            else:
                                response_text = "I couldn't generate a proper response for your query. Please try rephrasing your question."
                                
                            # Display response with typewriter effect
                            typewriter_response(message_placeholder, response_text)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", response_text)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                        except Exception as e:
                            error_msg = f"An error occurred while processing your request. Please try again."
                            st.warning(f"Error details: {str(e)}")
                            normal_response(message_placeholder, error_msg)
                            normal_response(message_placeholder, error_msg)
                            save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"I'm sorry, I encountered an error while processing your question. Please try again with a different question."
                st.warning(f"Error details: {str(e)}")
                normal_response(message_placeholder, error_msg)
                save_chat_to_history(st.session_state.current_chat_id, "assistant", error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()