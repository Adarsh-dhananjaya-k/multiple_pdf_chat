import streamlit as st
from keybert import KeyBERT
import PyPDF2
import os
import tempfile
import time
import logging
import psutil
import altair as alt
import pandas as pd

logging.basicConfig(filename='app.log', level=logging.INFO)

# Function to simulate metric data
def simulate_metric_data():
    return {
        'CPU Usage': psutil.cpu_percent(),
        'Memory Usage': psutil.virtual_memory().percent,
        'Network Activity': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
    }

def log_system_metrics(metrics):
    logging.info(f'CPU Usage: {metrics["CPU Usage"]}% | Memory Usage: {metrics["Memory Usage"]}% | Network Activity: {metrics["Network Activity"]} bytes')

def simulate_function():
    time.sleep(5)  # Simulate a task that takes 5 seconds to execute

# Function to send alert notification (replace with your notification logic)
def send_alert_notification(message):
    st.error(message)

def extract_text_from_pdf(uploaded_file):
    # Save the uploaded PDF to a temporary file
    temp_file_path = save_temporary_file(uploaded_file)

    text = ""
    with open(temp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    # Clean up: Remove the temporary file
    os.remove(temp_file_path)

    return text

def save_temporary_file(uploaded_file):
    # Save the uploaded file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    return temp_file_path

def extract_keywords_keybert(text):
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
    return keywords

def main():
    st.session_state.endtime=st.empty()
    st.title("Keyphrase Extractor App from Multiple PDFs")
    st.write(
        "This app extracts keyphrases from multiple PDF files using Hugging Face Transformers (keybert) and PyPDF2."
    )

    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    start_time = time.time()
    if uploaded_files:
        st.subheader("Results:")
        start_time = time.time()
        for uploaded_file in uploaded_files:

            # Extract text from the uploaded PDF file
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Extract keyphrases using the extract_keywords_keybert function
            keyphrases = extract_keywords_keybert(pdf_text)

            # Display the extracted keyphrases with weights
            st.subheader(f"Extracted Keyphrases from {uploaded_file.name}:")
            for keyword in keyphrases:
                st.write(f"{keyword[0]} (Weight: {keyword[1]:.4f})")
            
            st.markdown("---")  # Add a separator between results
        
        st.session_state.endtime=time.time()-start_time
    # Streamlit App
    with st.sidebar:
        st.subheader("Performance Monitoring")

        # Set a threshold for CPU usage
        cpu_threshold = 90

        # Initialize data for KPIs and historical data
        cpu_kpi_placeholder = st.empty()
        memory_kpi_placeholder = st.empty()
        network_kpi_placeholder = st.empty()
        execution_time_kpi = st.metric("Execution Time", help="Time taken to execute the function", value=0)

        # Streamlit App Logic
        start_time = time.time()

        while time.time() - start_time <= 60:  # Run for 60 seconds
            # Get system metrics
            metrics = simulate_metric_data()

            # Log metrics
            log_system_metrics(metrics)

            # Update KPIs
            cpu_kpi_placeholder.text(f'CPU Usage: {metrics["CPU Usage"]}%')
            memory_kpi_placeholder.text(f'Memory Usage: {metrics["Memory Usage"]}%')
            network_kpi_placeholder.text(f'Network Activity: {metrics["Network Activity"]} bytes')

            # Execute a function and measure its execution time
            start_function_time = time.time()
            print(st.session_state.endtime)
            # execution_time = time.time() - start_function_time
            execution_time_kpi.metric("Execution Time", value=st.session_state.endtime)

            # Raise an alert if CPU usage exceeds the threshold
            if metrics['CPU Usage'] > cpu_threshold:
                alert_message = f'High CPU Usage Alert! CPU: {metrics["CPU Usage"]}%'
                logging.error(alert_message)
                send_alert_notification(alert_message)

            # Simulate real-time updates
            time.sleep(1)

if __name__ == "__main__":
    main()
