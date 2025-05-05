import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import numpy as np  
from langchain.schema import Document

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

with open("vectorstore.pkl", "rb") as f:
    index, texts, metadata = pickle.load(f)

embedding_model = OpenAIEmbeddings()
chat = ChatOpenAI(temperature=0, openai_api_key=openai_key)
qa_chain = load_qa_with_sources_chain(chat, chain_type="stuff")

st.title("🤖 Ask Me Anything About Mariam")

query = st.text_input("Your question:")

if query:
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(query_embedding, k=4)
    source_texts = [Document(page_content=texts[i], metadata=metadata[i]) for i in I[0]]

    response = qa_chain({"input_documents": source_texts, "question": query}, return_only_outputs=True)

    st.markdown("### Answer")
    st.write(response["output_text"])

    st.markdown("### Sources")
    for i in I[0]:
        st.markdown(f"**{metadata[i]['source']}**\n> {texts[i][:200]}...")
