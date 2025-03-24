# Srisawas ChatBot Demo 

## Install Prerequisites

* python version 3.11.9

* pip install -r requirements.txt

## Running the application

* `streamlit run app.py`

## Tools 
* Chatbot Interface : สร้างส่วนติดต่อผู้ใช้ (UI) สำหรับแชทบอทด้วย Streamlit

* File Upload : ให้ผู้ใช้อัปโหลดเอกสาร (เช่น PDF, TXT) เพื่อเป็นแหล่งข้อมูลเพิ่มเติมได้

* Document Parsing : ใช้ LlamaParse เพื่อแยก Txt จากเอกสารหลายรูปแบบ

* Vector Embeddings : สร้าง vector embeddings ของ Txt โดยใช้ HuggingFace model (BAAI/bge-m3)

* Vector Database : ใช้ FAISS เพื่อสร้างและจัดเก็บ vector embeddings ทำให้ค้นหาข้อมูลที่เกี่ยวข้อง

* Retrieval-Augmented Generation (RAG) : ใช้ LangChain's RetrievalQA chain เพื่อ retrieval จาก vector database และ generation โดย LLM

* Large Language Model (LLM) : ใช้ LLM [(typhoon-v2-70b-instruct)](https://docs.opentyphoon.ai/) จาก opentyphoon.ai ในการสร้างคำตอบ

* Conversation Memory : ใช้ ConversationBufferMemory เพื่อให้ chatbot จดจำบริบทของการสนทนาก่อนหน้าได้

* Prompt Engineering : ใช้ PromptTemplate เพื่อกำหนด prompt ที่ชัดเจนและมีโครงสร้าง ช่วยให้ LLM ตอบคำถามได้ตรงประเด็นมากขึ้น