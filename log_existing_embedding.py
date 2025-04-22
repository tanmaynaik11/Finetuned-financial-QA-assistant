import mlflow
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Paths
VECTOR_DB_DIR = "chroma_tesla_db"
DB_FILE = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")

# Load Chroma to count chunks
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)
chunk_count = vectordb._collection.count()

# === MLflow Logging ===
mlflow.set_experiment("tesla-rag")

with mlflow.start_run(run_name="embedding_run_existing"):
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("chunk_size", 500)         # if known
    mlflow.log_param("chunk_overlap", 100)      # if known
    mlflow.log_metric("chunks_created", chunk_count)
    mlflow.log_artifact(DB_FILE)

print(f"✅ Existing ChromaDB run logged to MLflow with {chunk_count} chunks.")
