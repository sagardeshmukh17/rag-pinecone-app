# 🏢 Company RAG Chatbot (ChromaDB & Pinecone)

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **OpenAI embeddings**, and vector databases (**ChromaDB** and **Pinecone**).  
This app allows users to ask questions about company documents and get AI‑powered answers.


## 🚀 Features
- Upload and process company documents
- Generate embeddings using OpenAI
- Store vectors in **ChromaDB** (local) or **Pinecone** (cloud)
- Query documents with semantic search
- Interactive chatbot UI built with **Streamlit**


## 📂 Project Structure
├── app.py              # Streamlit UI
├── ingest.py           # Embedding + Vector storage
├── rag.py              # Query logic
├── documents/          # Company text files
├── requirements.txt    # Dependencies
└── .gitignore          # Ignore sensitive files

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/<username>/rag-pinecone-app.git
   cd rag-pinecone-app

2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Create .env file
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here

5. Run ingestion script
python ingest.py

6. Start Streamlit app
streamlit run app.py

7. Open browser at:
http://localhost:8501
