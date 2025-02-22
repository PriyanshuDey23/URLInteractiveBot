import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import *

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Extract text data from a URL
def extract_url_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Error extracting text from {url}: {e}")
        return ""

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Store chunks in vector
def get_vector_store(chunks):
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Create the QA chain
def get_conversational_chain():
    prompt_template = PROMPT
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process user query
def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "user_question": user_question}, return_only_outputs=True)
    
    with st.expander("🔍 View Response"):
        st.success(response["output_text"])

# Streamlit App
st.set_page_config(page_title="Chat with Multiple URLs", layout="wide")

st.title("🤖 Chat with Multiple URLs")
st.markdown("Ask AI questions based on extracted content from multiple URLs.")

# Sidebar for URL input
with st.sidebar:
    st.header("📌 Menu")
    urls = st.text_area("Enter URLs (one per line)", height=120)
    process_btn = st.button("🔄 Process URLs")

# Process URLs and store text
if process_btn:
    with st.spinner("🔍 Extracting and processing URLs..."):
        all_chunks = []
        for url in urls.splitlines():
            raw_text = extract_url_text(url)
            if raw_text.strip():
                all_chunks.extend(get_text_chunks(raw_text))
            else:
                st.warning(f"⚠️ No extractable text found in {url}.")
        if all_chunks:
            get_vector_store(all_chunks)
            st.success("✅ URLs processed successfully!")
        else:
            st.warning("⚠️ No valid content found in the provided URLs.")



# User inputs question
user_question = st.text_input("💬 Ask a Question")

# Display the "Answer" button only when a question is asked
if st.button("📝 Answer"):
    if user_question:
        user_input(user_question)
