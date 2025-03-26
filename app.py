import streamlit as st
import os
import nest_asyncio
import time
import requests
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

#ตัวอย่างข้อมูลที่เก็บไว้ สามารถแทรก CSV ได้
image_data = {
    "ขั้นตอน": "https://www.sawad.co.th/wp-content/uploads/2024/10/452800239_896789245826573_6595247655261158306_n-819x1024.jpg",
    "ขอเพิ่มวงเงินออนไลน์": "https://scontent.fbkk9-2.fna.fbcdn.net/v/t39.30808-6/486686347_1076360731202756_168808874607018632_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=127cfc&_nc_ohc=uJHbtZIQ3GYQ7kNvgHF4yWW&_nc_oc=AdlPyVfh41AJJVvHcdKikzqXRYeZcfKfAd7PBmC9TDVqDaLXMZ6ht6haqCQqphm58hn5mYtcDqqeXRGhjQYP5ORj&_nc_zt=23&_nc_ht=scontent.fbkk9-2.fna&_nc_gid=gJimbS1Yuc9gZTbg3EWpqA&oh=00_AYEUgeZgzkddThejgHiv-SW7TA0cvTKph9ngdIqKAc1cWA&oe=67E8B68B",
    "เอกสาร": "https://scontent.fbkk12-5.fna.fbcdn.net/v/t1.6435-9/49372192_1985324158202926_1335522292599357440_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=rZze6y4lyHwQ7kNvgExWIqY&_nc_oc=AdnR5Cx9QKLZmEB6VJ8vMwWqyhrZL5kqyxu-3S0zmn3XGK8evwrKL0WaCWASxEPkVzINNLD2hXI0LCvDpO9XazjC&_nc_zt=23&_nc_ht=scontent.fbkk12-5.fna&_nc_gid=8hNFyuKaJw90Gdkr7oa06g&oh=00_AYEhL4EmzLra2C01JcDkjDRB4sz4bwWdD1G7Yi2eGrfI8g&oe=680A2CB5",
    "ประกันภัยรถยนต์": "https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/486135644_1074484228057073_8174681586289252031_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=127cfc&_nc_ohc=5Dh2aGLdmMoQ7kNvgHfvWgs&_nc_oc=AdlWrWObPq0uusPLZeKFc4PaAttTPJPAp-Xf7mCbCrC2nClYldVN7MCP82r7E4tvibJ2IHQmJ7cBtKS-GxL2pT2J&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=NLRQU4IaSV8ZqRE4bnl37g&oh=00_AYHqVxlhwnfqdZYK82aAIXlDdE4GZSW7dCTgo8Yraj1h3w&oe=67E8ADAA"
}

model = SentenceTransformer("BAAI/bge-m3")
stored_texts = list(image_data.keys())
stored_embeddings = model.encode(stored_texts)

def find_best_match(user_input):
    input_embedding = model.encode([user_input])[0]
    similarities = np.dot(stored_embeddings, input_embedding) / (np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(input_embedding))
    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]
    SIMILARITY_THRESHOLD = 0.6
    
    if best_similarity >= SIMILARITY_THRESHOLD:
        best_match = stored_texts[best_index]       
        return image_data.get(best_match, None)
    return None

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

def create_chatbot_with_references(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={'k': 3}) if vector_db else None
    template = """
    You are an AI assistant specializing in providing information about SriSawad Company. 
    Use the following context to answer the question.
    Always cite your sources by adding the document name and line number at the end of each relevant piece of information.
    If the input is in any language, respond in that language. and name the document and line number.
    
    If you don't know the answer, say "I don't know."
    
    Context: {context}
    Question: {question}
    
    Answer (with references):
    """
    
    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    def process_response_with_references(response, query):
        docs = response['source_documents']
        references = []
        image_url = find_best_match(query)
        answer = response['result']
        if references:
            answer += "\n\nReferences:" + "\n".join(references)
        if image_url:
            answer = f"![Relevant Image]({image_url})\n\n{answer}"
            
        return answer

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type_kwargs={
            "prompt": PROMPT
        },
        return_source_documents=True
    ) if retriever else None
    
    if qa_chain:
        def wrapped_qa(query):
            raw_response = qa_chain(query)
            return {"result": process_response_with_references(raw_response, query["query"])}
        return wrapped_qa
    return None

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
            chatbot = create_chatbot_with_references(vector_db)
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
            if response.startswith("![Relevant Image]"):
                parts = response.split("\n\n", 1)
                if len(parts) == 2:
                    image_part, text_part = parts
                    message_placeholder.markdown(image_part)
                    text_placeholder = st.empty()
                    for i in range(len(text_part)):
                        text_placeholder.markdown(text_part[:i+1])
                        time.sleep(0.02)
                    full_response = response
            else:
                full_response = response
                for i in range(len(full_response)):
                    message_placeholder.markdown(full_response[:i+1])
                    time.sleep(0.02)
                    
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()