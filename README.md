# 🚗 Fine-tuned Financial QA Assistant — RAG + LoRA + FastAPI + Streamlit

An end-to-end **Retrieval-Augmented Generation (RAG)** system for answering questions about Tesla's SEC 10-K/10-Q filings using:

- 🔍 **ChromaDB** for document retrieval
- 🧠 **LoRA-fine-tuned TinyLLaMA** for custom domain knowledge
- ⚡ **Redis** for ultra-fast answer caching
- 🌐 **FastAPI** for programmatic access
- 🎨 **Streamlit** for interactive UI

---

# Introduction
The  AI Assistant is a full-stack, intelligent question-answering system purpose-built for analyzing  SEC filings using modern Generative AI techniques. It combines the power of:

🔍 Retrieval-Augmented Generation (RAG)

🧠 LoRA-fine-tuned LLM (TinyLLaMA-1.1B)

💽 ChromaDB vector search

⚡ Redis caching

🌐 FastAPI backend

🎨 Streamlit frontend

This assistant is capable of answering complex financial and regulatory questions by grounding responses in  actual 10-K/10-Q disclosures. For generic finance questions or when retrieval fails, the system falls back to a fine-tuned LLM trained on over 500 curated financial Q&A examples.

Whether you are:

A developer exploring end-to-end LLM architectures,

A finance analyst digging through corporate reports,

Or a researcher experimenting with domain-specific QA systems—

This project demonstrates a complete pipeline from data collection to deployment, blending MLOps, NLP, and LLM fine-tuning into one coherent system.

## 🧠 What It Does

Ask natural language questions like:

- “What are Tesla’s key risk factors?”
- “What does Tesla say about supply chain issues?”
- “What is compound interest?” (fallback → LoRA)

And get instant, grounded answers powered by a combination of retrieval + generative AI.

---

