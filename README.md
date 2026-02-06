# RAG Chatbot

An offline Retrieval-Augmented Generation (RAG) system that reads, indexes, and answers questions from multiple PDF documents using local LLaMA GGUF models.  
Runs fully locally — no API keys, no OpenAI, no cloud services.

---

## 1. Overview

This project implements a complete **local AI document assistant** capable of:

- Loading multiple PDFs  
- Extracting and cleaning text  
- Splitting text into semantic chunks  
- Generating vector embeddings  
- Indexing them using FAISS  
- Retrieving relevant context for any query  
- Generating high-quality answers through a local LLaMA model  

It closely replicates ChatGPT-style RAG systems while running **entirely offline** on your machine.

---

## 2. Features

### Multi-PDF Support
Load, read, and process any number of PDFs inside the `data/` directory.

### Text Chunking
Optimized chunk splitting (800 chars with overlap) for accurate retrieval.

### Embeddings
Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, lightweight embeddings.

### FAISS Vector Store
Efficient dense vector search over thousands of chunks.

### Local LLaMA GGUF Inference
Runs LLaMA 3.1 8B Instruct in GGUF format using `llama-cpp-python`.

### Streamlit Chat UI
Includes:

- Chat bubbles  
- Clean, minimal design  
- Model status display  
- Multi-PDF retrieval support  

### Offline RAG Pipeline
No internet connection required.

---

## 3. Project Structure
ChatGPT said:

Here is the entire README, complete, professional, clean, and in full copy-paste code block format.
Just copy and replace your current README.md.

# Local Multi-PDF LLaMA Assistant

An offline Retrieval-Augmented Generation (RAG) system that reads, indexes, and answers questions from multiple PDF documents using local LLaMA GGUF models.  
Runs fully locally — no API keys, no OpenAI, no cloud services.

---

## 1. Overview

This project implements a complete **local AI document assistant** capable of:

- Loading multiple PDFs  
- Extracting and cleaning text  
- Splitting text into semantic chunks  
- Generating vector embeddings  
- Indexing them using FAISS  
- Retrieving relevant context for any query  
- Generating high-quality answers through a local LLaMA model  

It closely replicates ChatGPT-style RAG systems while running **entirely offline** on your machine.

---

## 2. Features

### Multi-PDF Support
Load, read, and process any number of PDFs inside the `data/` directory.

### Text Chunking
Optimized chunk splitting (800 chars with overlap) for accurate retrieval.

### Embeddings
Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, lightweight embeddings.

### FAISS Vector Store
Efficient dense vector search over thousands of chunks.

### Local LLaMA GGUF Inference
Runs LLaMA 3.1 8B Instruct in GGUF format using `llama-cpp-python`.

### Streamlit Chat UI
Includes:

- Chat bubbles  
- Clean, minimal design  
- Model status display  
- Multi-PDF retrieval support  

### Offline RAG Pipeline
No internet connection required.

---

## 3. Project Structure



local-multi-pdf-llama-assistant/
│
├── app.py # Streamlit app (transformers)
├── app_chat_gguf.py # Chat UI with GGUF model
├── app_gguf.py # GGUF-based simple UI
│
├── modules/
│ ├── pdf_loader.py
│ ├── multi_pdf_loader.py
│ ├── text_splitter.py
│ ├── embedder.py
│ ├── vector_store.py
│ ├── pipeline.py
│ ├── multi_rag.py
│ ├── local_llm.py
│ ├── local_llm_gguf.py
│
├── models/
│ └── llm.gguf # LLaMA model (not uploaded)
│
├── data/
│ └── sample.pdf
│
├── requirements.txt
├── README.md
├── .gitignore


---

## 4. System Architecture
             ┌────────────────────────┐
             │        PDF Files        │
             │      (multiple PDFs)    │
             └──────────────┬─────────┘
                            │
             ┌──────────────▼─────────────┐
             │         PDF Loader          │
             │ (extract & clean text)      │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │       Text Splitter         │
             │ (chunking with overlap)     │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │       Embedder (MiniLM)     │
             │  (vector representations)    │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │        FAISS Index          │
             │ (fast similarity search)    │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │    Retriever + Prompt       │
             │  (top-k context selection)  │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │  Local LLaMA Model (GGUF)   │
             │    via llama-cpp-python     │
             └──────────────┬─────────────┘
                            │
             ┌──────────────▼─────────────┐
             │        Final Answer         │
             └─────────────────────────────┘


---

## 5. Installation

### Step 1 — Clone
git clone https://github.com/swarajshinde12/local-multi-pdf-llama-assistant

cd local-multi-pdf-llama-assistant


### Step 2 — Create Virtual Environment

python -m venv venv
.\venv\Scripts\activate


### Step 3 — Install Dependencies



pip install -r requirements.txt


### Step 4 — Download GGUF Model

Download the file:



Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf


Place it in:



models/llm.gguf


---

## 6. Running the Application

### Option A — Chat UI with GGUF



streamlit run app_chat_gguf.py


### Option B — Simple GGUF App



streamlit run app_gguf.py


### Option C — Transformers Pipeline



streamlit run app.py


### Option D — Test Model Only



python modules/local_llm_gguf.py


---

## 7. How the RAG Pipeline Works

1. Load PDFs  
2. Extract + clean text  
3. Split text into overlapping chunks  
4. Generate MiniLM embeddings  
5. Build FAISS index  
6. Embed user query  
7. Retrieve top-k relevant chunks  
8. Feed them into LLaMA for contextual answer generation  

This ensures high-quality offline question answering across multiple documents.

---

## 8. Performance Notes

- CPU-only works but slower.
- GPU acceleration via llama-cpp improves generation speed significantly.
- Embedding model loads once and handles thousands of chunks efficiently.
- Suitable for academic research, enterprise offline use, and personal projects.

---

## 9. Future Enhancements

- Chat memory  
- Persistent FAISS index storage  
- GPU-accelerated embeddings  
- UI redesign with sidebar file previews  
- Model switching from interface  

---

## 10. License

MIT License.

---

## 11. Author

**Swaraj Vijay Shinde**  
Final Year B.Tech – Data Science  
Creator of the Local Multi-PDF LLaMA Assistant

---




