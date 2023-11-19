import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings,GooglePalmEmbeddings
from langchain.vectorstores import FAISS,Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
from htmlTemplates import css,bot_template,user_template

def get_pdf_test(pdf_docs):
    text = ""
    for pdf in pdf_docs:
         pdf_reader = PdfReader(pdf)
         for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_raw_chunck(raw_text):
     test_splitter=CharacterTextSplitter( separator="\n",
                                         chunk_size=1000,
                                         chunk_overlap=200,
                                         length_function=len)
     chunks=test_splitter.split_text(raw_text)
     return chunks
  
def get_vectorStore(text_chunks):


    st.write("hello")
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings=GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = GooglePalm(temperature=1.0)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history=response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
   
    st.set_page_config(page_title="chats with muiltple PDFs",page_icon="books")

    st.write(css,unsafe_allow_html=True)

   

    if "conversion" not in st.session_state:
        st.session_state.conversion=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with muiltple pdf's :books")
    user_question=st.text_input("ask a question about your documnets ")

    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("your documents")
        pdf_doc=st.file_uploader("upload your PDF here ", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                raw_text=get_pdf_test(pdf_doc)
                

                text_chuckes=get_raw_chunck(raw_text)
                

                vector_store=get_vectorStore(text_chuckes)

                st.session_state.conversation = get_conversation_chain(
                    vector_store)
                





if __name__=='__main__':
    main()
