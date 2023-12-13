import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings,GooglePalmEmbeddings
from langchain.vectorstores import FAISS,Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm,HuggingFaceHub
from htmlTemplates import css,bot_template,user_template
from googletrans import Translator
import time
import logging
import psutil 

def translate_text(text, target_language='en'):
    print("tranlate in text to ",target_language)
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def get_pdf_test(pdf_docs):
    text = ""
    for pdf in pdf_docs:
         pdf_reader = PdfReader(pdf)
         for page in pdf_reader.pages:
            text += page.extract_text()
    print("pdf ext")
    return text

def get_raw_chunck(raw_text):
     test_splitter=CharacterTextSplitter( separator="\n",
                                         chunk_size=1000,
                                         chunk_overlap=200,
                                         length_function=len)
     chunks=test_splitter.split_text(raw_text)
     print("done dhunks")
     return chunks
  
def get_vectorStore(text_chunks):

    # st.write("hello")
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings=GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("done embedding")
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = GooglePalm(temperature=0.4)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print("done conversion chain")

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': translate_text(user_question,"en")})
    st.session_state.chat_history=response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
            if st.session_state.target_language =='en':
                if i%2==0:
                        st.write(user_template.replace(
                            "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}",message.content), unsafe_allow_html=True)
            
            else :
                if i%2==0:
                        st.write(user_template.replace(
                            "{{MSG}}", translate_text(message.content,st.session_state.target_language)), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", translate_text(message.content,st.session_state.target_language)), unsafe_allow_html=True)
    
            





def main():
    
    load_dotenv()
    indian_languages = {
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'marathi': 'mr',
    'tamil': 'ta',
    'urdu': 'ur',
    'gujarati': 'gu',
    'malayalam': 'ml',
    'kannada': 'kn',
    'oriya': 'or',
    'punjabi': 'pa',
    'assamese': 'as',
    'maithili': 'mai',
    'santali': 'sat',
    'kashmiri': 'ks',
    'konkani': 'kok',
    'sindhi': 'sd',
    'nepali': 'ne',
    'dogri': 'doi',
    'bodo': 'brx',
    'khasi': 'kha',
    'mizo': 'lus',
    'garo': 'grt',
    'manipuri': 'mni',
    'kokborok': 'trp',
    'english': 'en'
}

   
    st.set_page_config(page_title="chats with muiltple PDFs",page_icon="books")
    st.title("Main Page")
    st.write(css,unsafe_allow_html=True)

   

    if "conversion" not in st.session_state:
        st.session_state.conversion=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with muiltple pdf's :books")
    user_question=st.text_input("ask a question about your documnets ")

    if user_question:
        try:
            handle_userinput(user_question)
        except:
            st.write("upload file")

            
    st.session_state.endtime=time.time()
    with st.sidebar:
        
        Start_time = time.time()
        st.subheader("your documents")
        pdf_doc=st.file_uploader("upload your PDF here ", accept_multiple_files=True)
        st.session_state.page1_file = pdf_doc
        Start_time = time.time()
        if st.button("process"):
            with st.spinner("processing"):
                raw_text=get_pdf_test(pdf_doc)
                

                text_chuckes=get_raw_chunck(raw_text)
                

                vector_store=get_vectorStore(text_chuckes)

                st.session_state.conversation = get_conversation_chain(
                    vector_store)
                
        st.session_state.endtime=time.time()-Start_time        
        
        print("endtime",st.session_state.endtime)
        language=st.selectbox("Select target language:",['english','hindi', 'bengali', 'telugu', 'marathi', 'tamil', 'urdu', 'gujarati', 'malayalam', 'kannada', 'oriya', 'punjabi', 'assamese', 'maithili', 'santali', 'kashmiri', 'konkani', 'sindhi', 'nepali', 'dogri', 'bodo', 'khasi', 'mizo', 'garo', 'manipuri'])        
        st.session_state.target_language =indian_languages[language]
        print(st.session_state.target_language)
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
        kpi4.metric(label="time", value= f"{round(time)} ")
        st.write(f'CPU Usage: {cpu}% | Memory Usage: {memory}% | Network Activity: {network.bytes_sent + network.bytes_recv} bytes')

        # Raise an alert if CPU usage exceeds the threshold
        if cpu > cpu_threshold:
            alert_message = f'High CPU Usage Alert! CPU: {cpu}%'
            logging.error(alert_message)
            send_alert_notification(alert_message)






if __name__=='__main__':
    main()
