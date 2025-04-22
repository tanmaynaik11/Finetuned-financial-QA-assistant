from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import redis
import hashlib

# --- Redis Setup ---
try:
    rdb = redis.Redis(host='localhost', port=6379, db=0)
    rdb.ping()  # Check if Redis server is running
    print("✅ Connected to Redis")
except redis.exceptions.ConnectionError:
    print("⚠️ Redis server not available. Caching disabled.")
    rdb = None

def get_cache_key(query):
    """Use a hash of the query string to prevent length issues."""
    return f"qa:{hashlib.sha256(query.encode()).hexdigest()}"

# === Step 1: Load Embeddings + VectorDB ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_tesla_db", embedding_function=embedding_model)

# === Step 2: Load Fine-Tuned TinyLLaMA Model ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, "tinyllama_lora_adapter")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# === Step 3: RAG Inference Function ===
def generate_rag_answer(query, k=3, score_threshold=0.8):

    # Step 0: Check Redis cache
    key = get_cache_key(query)
    cached_answer = rdb.get(key) if rdb else None
    if cached_answer:
        print("⚡ Retrieved from Redis cache\n")
        return cached_answer.decode("utf-8")
    
    # Step 1: Search ChromaDB
    docs_and_scores = vectordb.similarity_search_with_score(query, k=k)

    good_docs = [doc for doc, score in docs_and_scores if score < score_threshold]

    # Step 2: Join docs to build context
    context = "\n".join([doc.page_content for doc in good_docs]).strip()

    # Step 3: Fallback check + print status
    if not good_docs:
        print("🤖 [Fallback: Answering without retrieval]\n")
        prompt = f"### Question: {query}\n### Answer:"
    else:
        print("🔍 [Answering with retrieved context]\n")
        prompt = f"""### Context:
{context}

### Question: {query}
### Answer:"""

    # Step 4: Run model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if rdb:
        rdb.set(key, answer)
    return answer

# === Step 4: CLI Loop ===
if __name__ == "__main__":
    while True:
        q = input("📩 Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n" + generate_rag_answer(q))
        print("—" * 60)
