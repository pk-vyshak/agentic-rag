# 📑 Agentic-RAG

A **Streamlit-based AI assistant** that helps users search and query PDF documents using Retrieval-Augmented Generation (RAG). It uses Qdrant as a vector store, SentenceTransformers for embeddings, and the **Agno agentic framework** to manage reasoning and tool orchestration with Azure OpenAI models.

---

## 🚀 Features

- Upload and index any PDF document
- Ask natural language questions about the document
- Retrieves relevant paragraphs using vector search (Qdrant)
- Uses **Agno agents** to decompose complex queries and route them to reasoning tools
- Generates accurate answers with confidence scores
- Built-in retry logic and robust fallback handling

---

## 🧱 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Agent Framework**: [Agno](https://github.com/agnos-ai/agno)
- **Vector DB**: Qdrant
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Azure OpenAI GPT (configurable)

---

## 📦 Setup Instructions

### 1. 🔧 Clone the Repository

```bash     
# Clone the repository
git clone https://github.com/your-username/document-assistant.git
cd document-assistant

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate     # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (requires Docker)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Run the Streamlit app
streamlit run src/app.py