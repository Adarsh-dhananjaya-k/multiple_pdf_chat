import streamlit as st
import time
import logging
import psutil  # For system metrics

# Configure logging
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

# st.title("Performance Monitoring")
def perform_monitoring():
# Set a threshold for CPU usage
    cpu_threshold = 90
 
    placeholder = st.empty()
 
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        cpu, memory, network = get_system_metrics()

        # Log metrics
        log_system_metrics(cpu, memory, network)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="CPU Usage", value=cpu, delta= round(cpu) - 10)
        kpi2.metric(label="Memory Usage", value= int(memory), delta= - 10 + memory)
        kpi3.metric(label="Network Activity", value= f"{round(network.bytes_sent + network.bytes_recv)} ")
        st.write(f'CPU Usage: {cpu}% | Memory Usage: {memory}% | Network Activity: {network.bytes_sent + network.bytes_recv} bytes')

        # Raise an alert if CPU usage exceeds the threshold
        if cpu > cpu_threshold:
            alert_message = f'High CPU Usage Alert! CPU: {cpu}%'
            logging.error(alert_message)
            send_alert_notification(alert_message)



if __name__ == '__main__':
    perform_monitoring()