from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "BAAI/bge-m3"
SAVE_PATH = "/app/models"

model = SentenceTransformer(MODEL_NAME)
model.save(os.path.join(SAVE_PATH, MODEL_NAME.replace("/", "--")))