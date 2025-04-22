from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

# Paths
DATA_DIR = "tesla_sec_filings"
VECTOR_DB_DIR = "chroma_tesla_db"

# Step 1: Load and merge all filings
documents = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_DIR, fname), encoding="utf-8")
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents.")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")

# Step 3: Embed and store in Chroma
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_DIR
)

vectordb.persist()
print(f"✅ ChromaDB created at ./{VECTOR_DB_DIR}/ with {len(chunks)} entries.")
