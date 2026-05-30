# 🤖 RAG Chatbot — Gemini Embedding 2 Paper

A Retrieval-Augmented Generation chatbot built with **OpenAI GPT-4o Mini**, 
**OpenAI Embeddings**, and **Streamlit** — no external vector DB required.

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py            # Streamlit UI
├── data_loader.py    # PDF / CSV / Excel loader
├── vector_store.py   # In-memory embeddings + cosine search
├── rag_chain.py      # RAG pipeline (retrieve → prompt → answer)
├── requirements.txt  # Dependencies
└── data/
    └── 2605_27295v1.pdf   # Default dataset (Gemini Embedding 2 paper)
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. In the browser
1. Enter your **OpenAI API Key** in the sidebar
2. Choose **"Use default dataset"** (the PDF is pre-loaded) or upload your own files
3. Click **Build Knowledge Base** — this embeds all chunks via OpenAI
4. Start chatting!

---

## 🧠 How it Works

```
User Question
     │
     ▼
OpenAI Embeddings (text-embedding-3-small)
     │  embed the query
     ▼
Cosine Similarity Search over stored chunks
     │  retrieve top-K relevant passages
     ▼
Build prompt: System + Chat History + Context + Question
     │
     ▼
OpenAI GPT-4o Mini  →  Answer with source citations
```

---

## 📂 Supported File Types

| Format | How it's chunked |
|--------|-----------------|
| PDF    | One chunk per page |
| CSV    | 10 rows per chunk |
| Excel  | 10 rows per sheet chunk |

---

## ⚙️ Key Settings (sidebar)

| Setting | Default | Description |
|---------|---------|-------------|
| Top-K   | 5       | Number of chunks retrieved per query |
| Model   | gpt-4o-mini | OpenAI chat model |
| Embedding | text-embedding-3-small | OpenAI embedding model |

---

## 💡 Example Questions (for the default PDF)

- "What is Gemini Embedding 2?"
- "How does native audio outperform ASR?"
- "What benchmarks were used to evaluate the model?"
- "Explain the training recipe used."
- "What are the key differences from CLIP?"
