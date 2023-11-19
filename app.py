import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from PyPDF2 import PdfFileReader


def get_pdf_test(pdf_doc):
    test=""
    for pdf in pdf_doc:
        pdf_reader=PdfFileReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text



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
                raw_test=get_pdf_test(pdf_doc)





if __name__=='__main__':
    main()
