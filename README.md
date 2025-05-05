# Mariam's Personal LLM Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about my resume, GitHub projects, and any other personal documents I've uploaded. It uses OpenAI's GPT-3.5 Turbo and FAISS for semantic search.

## Features
- Ask questions about my resume or GitHub repos
- Retrieve content from my GitHub files (PDFs, Markdown, code comments) - TBD
- Summarize skills or projects
- Streamlit interface for interaction
- Jupyter Notebook with a simple GUI (text input + response display)

## Requirements
- Python 3.9+
- OpenAI API key

## Installation
### Step 1: Create a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

## Setup
1. Run the indexer:
```bash
python index_documents.py
```
2. Launch the Streamlit chatbot:
```bash
streamlit run app.py
```
3. Or use the Jupyter Notebook GUI:
```bash
jupyter notebook chatbot_gui.ipynb
```

## Example Questions
- "What is Mariam's experience with GNNs?"
- "What are Mariam's key technical skills?"