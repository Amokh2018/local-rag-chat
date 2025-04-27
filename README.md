
# Chat with Your Documents – Local RAG with LLMs

Interact privately with your local documents using **Retrieval-Augmented Generation (RAG)** and **open-source LLMs** running on your machine via [Ollama](https://ollama.com/).  
No cloud. No data leakage. Full transparency.

---

## Features

- Load and chunk your own PDFs
- Embed with `sentence-transformers`
- Build a FAISS vector index
- Query documents using a local LLM like **Mistral** or **LLaMA2**
- Get references used to answer


---

## Project Structure

```plaintext
local-rag-chat/
├── data/documents/         # Input documents (PDFs)
├── embeddings/             # FAISS index and chunk cache
├── llm/                    # Prompt templates
├── rag/                    # RAG logic       
├── streamlit_app.py        # Web UI using Streamlit
├── notebook.ipynb          # Jupyter notebook demo
├── requirements.txt        # Dependencies
├── Makefile                # Common commands
└── README.md               # You're here!
```

---

## ⚙️ Setup

### 1. Create Environment & Install Dependencies

```bash
make init
source .venv/bin/activate
```

### 2. Add Your Documents

Put your `.pdf` files inside the `data/documents/` folder.

### 3. Build the Index

```bash
make index
```

### 4. Start Chatting

#### Web App:

```bash
make run
```

---

### Example Prompt Template

```text
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say so.

Context:
{context}

Question:
{question}
```

You can customize it in: `llm/prompt_template.txt`

---

### Demo Notebook

Check [`notebook.ipynb`](notebook.ipynb) for a step-by-step walkthrough:  
📄 Load PDFs → ✂️ Chunk → 🧠 Embed → 🔍 Search → 💬 Ask

---

### Environment Variables

Create a `.env` file or use `.env.example` to set:

```env
OLLAMA_MODEL=mistral
```

---

### Useful Make Commands

```bash
make init      # Create venv and install dependencies
make index     # Build FAISS index from PDFs
make run       # Launch Streamlit app
```

---

### 100% Private & Local

This tool is designed to **keep your data on your machine**.  
Perfect for confidential documents, research papers, internal reports, or teaching demos.

---

## 📬 Contact

Created by **Ali Mokh**  
Feel free to reach out for questions or improvements!
- Email: ali.mokh@ericsson.com


