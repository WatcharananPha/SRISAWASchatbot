FROM python:3.11.9

ARG MODEL_NAME="BAAI/bge-m3"

ARG SAFE_MODEL_NAME="BAAI--bge-m3"

ARG MODEL_PATH="/app/models"

WORKDIR /app

ENV LOCAL_MODEL_DIR=${MODEL_PATH}/${SAFE_MODEL_NAME}
RUN mkdir -p ${LOCAL_MODEL_DIR}

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; print(f'Downloading model ${MODEL_NAME}...'); model = SentenceTransformer('${MODEL_NAME}'); print(f'Saving model to ${LOCAL_MODEL_DIR}...'); model.save('${LOCAL_MODEL_DIR}'); print('Model saved.')"

COPY . /app

RUN mkdir -p /root/.streamlit

RUN cp config.toml /root/.streamlit/config.toml # แก้ไข typo จาก confaig.toml

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]