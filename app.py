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
from prompt import  * 


# Load environment variables from .env file
load_dotenv()

# Set up the API key for Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embedding model once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Extract text data from a URL:
def extract_url_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {url}: {e}")
        return ""

# Divide the text into chunks
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Store the chunks in vector
def get_vector_store(chunks):
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Create the qa chain
def get_conversational_chain():
    prompt_template = PROMPT # Imported

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.2)
    
    # Make sure context and user question  is included in the input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
    
    # Create the QA chain using the prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input
def user_input(user_question):
    
        # Load DB
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        
        # Get the conversational chain
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "user_question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])


# Create the Streamlit app
def main():
    st.set_page_config(page_title="Chat with Multiple URLs")
    st.header("Chat with Multiple URLs")

    # Input prompt for the user
    user_question = st.text_input("Ask the Question.. ", key="input")

    if user_question:
        user_input(user_question)

    # Sidebar to input multiple URLs
    with st.sidebar:
        st.title("Menu:")
        
        # Allow multiple URL inputs
        urls = st.text_area("Enter your URLs (one per line)", height=100)
        
        if urls and st.button("Submit and Process"):
            with st.spinner("Loading..."):
                all_chunks = []  # To store all text chunks from all URLs
                
                # Loop through each URL
                for url in urls.splitlines():
                    raw_text = extract_url_text(url)
                    if raw_text.strip() == "":
                        st.warning(f"URL {url} contains no extractable text.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        all_chunks.extend(text_chunks)  # Add chunks from this URL to the list
                
                if all_chunks:
                    get_vector_store(all_chunks)
                    st.success("Processing done for all URLs!")
                else:
                    st.warning("No valid text was extracted from the URLs.")

if __name__ == "__main__":
    main()