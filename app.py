import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings,GooglePalmEmbeddings
from langchain.vectorstores import FAISS,Pinecone


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
    query_result = embeddings.embed_query("Hello World")
    print("Length", len(query_result))
    st.write(len(query_result))
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # return vectorstore


def main():
    load_dotenv()

    st.set_page_config(page_title="chats with muiltple PDFs",page_icon="books")
    st.header("Chat with muiltple pdf's :books")
    st.text_input("ask a question about your documnets ")

    with st.sidebar:
        st.subheader("your documents")
        pdf_doc=st.file_uploader("upload your PDF here ", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                raw_text=get_pdf_test(pdf_doc)
                

                text_chuckes=get_raw_chunck(raw_text)
                

                vetor_store=get_vectorStore(text_chuckes)





if __name__=='__main__':
    main()
