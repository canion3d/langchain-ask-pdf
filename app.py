from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import gpt4all
import os
import streamlit_elements
from streamlit_elements import elements, mui, html
from quickstart import check_password
from quickstart import logout
from st_pages import Page, Section, add_page_title, show_pages, add_indentation, hide_pages

        st.write("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto');
        html, body, [class*="css"]  {
           font-family: 'Roboto'
        }
        </style>
        """, unsafe_allow_html=True)


gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")
messages = [{"role": "user", "content": "Name 3 colors"}]
gptj.chat_completion(messages)

os.environ["OPENAI_API_KEY"] = 


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask a PDF")
    st.header("Ask a PDF! 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF now to get quick answers for a PDF document!", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)


      from pydantic import BaseModel
      from typing import List
      from langchain.embeddings.openai import OpenAIEmbeddings

      class Input(BaseModel):
          text: str

      def generate_embeddings(inputs: List[Input]):
          openai_api_key =   # Replace with your actual OpenAI API key
          embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

      # Call your function with the necessary inputs

      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)

      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)

if __name__ == '__main__':
    main()

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
               <style>
               #MainMenu {visibility: hidden;}
               footer {visibility: hidden;}
               header {visibility: hidden;}
               </style>
               """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.balloons()
