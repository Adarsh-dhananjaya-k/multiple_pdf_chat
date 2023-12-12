from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.llms import GooglePalm
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
# from apikey import apikey
import os
import tempfile
import sys
sys.path.append('/Users/adarsh/Documents/multiple_pdf_chat/')
# from app.py import pdf_doc

# Set up OpenAI API
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDRnIO0C85J8vRZg8TBtQQLpNbDCUXmZdw"

# llm = OpenAI(temperature=0)
# llm=GooglePalm(temperature=1.0)
llm = LlamaCpp(
    streaming = True,
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    top_p=1, 
    verbose=True,
    n_ctx=4096
)
print(llm)

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        summaries.append(summary)

        # Delete the temporary file
        os.remove(temp_path)
    
    return summaries

# Streamlit App
st.title("Multiple PDF Summarizer")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        with st.spinner("processing"):
            st.write("Summaries:")
            summaries = summarize_pdfs_from_folder(pdf_files)
            for i, summary in enumerate(summaries):
                print(i,"llop")
                st.write(f"Summary for PDF {i+1}:")
                st.write(summary)