import os
import faiss
import pickle
import PyPDF2
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import numpy as np  

load_dotenv()

embedding_model = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

texts = []
metadata = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        with open(f"data/{file}", "rb") as f:
            reader = PyPDF2.PdfReader(f)
            raw_text = " ".join([page.extract_text() for page in reader.pages])
    else:
        with open(f"data/{file}", "r", encoding="utf-8") as f:
            raw_text = f.read()
    
    chunks = text_splitter.split_text(raw_text)
    texts.extend(chunks)
    metadata.extend([{"source": file}] * len(chunks))

embeddings = embedding_model.embed_documents(texts)
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings).astype("float32"))
with open("vectorstore.pkl", "wb") as f:
    pickle.dump((index, texts, metadata), f)

print("Documents indexed successfully.")
