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
import Performace
import streamlit as st
import time
import logging
import psutil 
# from apikey import apikey
import os
import tempfile
import sys
sys.path.append('/Users/adarsh/Documents/multiple_pdf_chat/')
# from app.py import pdf_doc

# Set up OpenAI API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDRnIO0C85J8vRZg8TBtQQLpNbDCUXmZdw"

# llm = OpenAI(temperature=0)
llm=GooglePalm(temperature=.4)
# The commented code `# llm = LlamaCpp(...)` is initializing an instance of the `LlamaCpp` class from
# the `langchain.llms` module. It is setting up the LlamaCpp language model with specific parameters
# such as streaming, model path, temperature, top_p, verbose, and n_ctx. These parameters control the
# behavior and performance of the LlamaCpp model.
# llm = LlamaCpp(
#     streaming = True,
#     model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     temperature=0.75,
#     top_p=1, 
#     verbose=True,
#     n_ctx=4096
# )
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
def main():
        st.title("Multiple PDF Summarizer")
        try:
        # Allow user to upload PDF files
            pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            
            if pdf_files:
                start_time = time.time()
                # Generate summaries when the "Generate Summary" button is clicked
                if st.button("Generate Summary"):
                    start_time = time.time()
                    with st.spinner("processing"):
                        st.write("Summaries:")
                        summaries = summarize_pdfs_from_folder(pdf_files)
                        for i, summary in enumerate(summaries):
                            print(i,"llop")
                            st.write(f"Summary for PDF {i+1}:")
                            st.write(summary)
        
            st.session_state.endtime=time.time()-start_time
        except:
            st.write("please upload the file  ")

        with st.sidebar:
            st.subheader("PERFORMANCE measurement")
            perform_monitoring(st.session_state.endtime)





logging.basicConfig(filename='app.log', level=logging.INFO)

# Function to get system metrics
def get_system_metrics():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    network_activity = psutil.net_io_counters()

    return cpu_usage, memory_usage, network_activity

# Function to log system metrics
def log_system_metrics(cpu, memory, network):
    logging.info(f'CPU Usage: {cpu}% | Memory Usage: {memory}% | Network Activity: {network}')

# Function to send alert notification (replace with your notification logic)
def send_alert_notification(message):
    st.error(message)

# Streamlit App



def perform_monitoring(time):
# Set a threshold for CPU usage
    cpu_threshold = 90
 
    placeholder = st.empty()
 
    with placeholder.container():
        kpi1, kpi2, kpi3,kpi4 = st.columns(4)
        cpu, memory, network = get_system_metrics()

        # Log metrics
        log_system_metrics(cpu, memory, network)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="CPU Usage", value=cpu, delta= round(cpu) - 10)
        kpi2.metric(label="Memory Usage", value= int(memory), delta= - 10 + memory)
        kpi3.metric(label="Network Activity", value= f"{round(network.bytes_sent + network.bytes_recv)} ")

        kpi4.metric(label="time", value= f"{time} ")
        st.write(f'CPU Usage: {cpu}% | Memory Usage: {memory}% | Network Activity: {network.bytes_sent + network.bytes_recv} bytes')

        # Raise an alert if CPU usage exceeds the threshold
        if cpu > cpu_threshold:
            alert_message = f'High CPU Usage Alert! CPU: {cpu}%'
            logging.error(alert_message)
            send_alert_notification(alert_message)

if __name__=='__main__':
        main()