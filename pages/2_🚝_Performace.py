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
st.title("Performance Monitoring")

# Set a threshold for CPU usage
cpu_threshold = 90

# Initialize KPI values
cpu_kpi = st.metric(label="CPU Usage", value=0, delta=0)
memory_kpi = st.metric(label="Memory Usage", value=0, delta=0 )
network_kpi = st.metric(label="Network Activity", value=0, delta=0)

# Streamlit App Logic
while True:
    # Get system metrics
    cpu, memory, network = get_system_metrics()

    # Log metrics
    log_system_metrics(cpu, memory, network)

    # Update KPI values
    cpu_kpi.value = cpu
    memory_kpi.value = memory
    network_kpi.value = network.bytes_sent + network.bytes_recv

    # Display metrics in Streamlit
    st.write(f'CPU Usage: {cpu}% | Memory Usage: {memory}% | Network Activity: {network.bytes_sent + network.bytes_recv} bytes')

    # Raise an alert if CPU usage exceeds the threshold
    if cpu > cpu_threshold:
        alert_message = f'High CPU Usage Alert! CPU: {cpu}%'
        logging.error(alert_message)
        send_alert_notification(alert_message)

    # Simulate real-time updates
    time.sleep(1)  # Adjust the sleep interval as needed
