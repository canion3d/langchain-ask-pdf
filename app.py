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

#All testing steps and process directions written by Michele Le, Ayoub El Bzioui, and Ty Canion of RBI GMbH#

#st.set_page_config(
#	layout="centered",
#	page_icon=":bar_chart:")

from streamlit_option_menu import option_menu

LoginImage = st.image('jrcm_logo_002.jpg', use_column_width=1)
LoginText = st.markdown("<h1 style='text-align: center; color: black;'>Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)


def Academy(st):
    if check_password() == True:
        add_indentation()
        hide_pages(["TCU_Testing", "SDS"])
        #global st # declare st as a global variable
        import streamlit as st
        LoginImage.empty()
        LoginText.empty()

        show_ul_style = """
        <style>
            ul { display: block !important; }
        </style>
        """
        # Call the markdown method to display the CSS style
        st.markdown(show_ul_style, unsafe_allow_html=True)

        # 1. horizontal menu

        selected = option_menu(None, ["Home", "Customers", "Management", "Tooling", "Incident Management", "RBI Academy", "Rexx Portal"],
            icons=['house', 'cloud-upload', 'menu-app', 'menu-app','menu-app', 'file-spreadsheet'],
            menu_icon="cast", default_index=0, orientation="horizontal",styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "15px"},
                "nav-link": {"font-size": "12px", "text-align": "left", "margin":"10px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "red"},
            }
        )

       # if "Home" == selected: os.startfile("C:\\Users\\Tyrone.Canion\\Desktop\\Projects\\RBI_App\\home.py")

        #if "Customers" == selected: open("C:\\Users\\Tyrone.Canion\\Desktop\\Projects\\RBI_App\\pages\\Customers.py")

        if selected ==  "Management" : os.startfile("C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Diagnostic Tool Set 8.16\DTS Monaco.lnk")

        if selected == "Tooling": os.startfile("C:\\Users\\Tyrone.Canion\\AppData\\Local\\Postman\\app-9.31.0\\postman.exe")

        if selected ==  "Incident Management" : os.startfile("https://gsep.daimler.com/jira/secure/RapidBoard"
                                                                         ".jspa?rapidView=75097&projectKey=TCUSQ&selectedIssue=TCUSQ-85")

        if selected == "Rexx Portal" : os.startfile("https://rbi-online.rexx-systems.com/login.php?goto=%2Fportal%2Frx%2F%3FtargetRealm%3Dportal%26menu%3D1%26script%3Dmy_workday%2Fmy_workday.php")

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

os.environ["OPENAI_API_KEY"] = "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"


def main():
    load_dotenv()
    st.set_page_config(page_title="JRC Mobility Ask a PDF")
    st.header("JRC Mobility - Ask a PDF! 💬")
    
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
          openai_api_key = "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"  # Replace with your actual OpenAI API key
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


st.markdown(
    "RBI Mission Statement: Through our experienced testers and technicians we qualify and maintain the customer"
    " systems and components to the highest satisfaction of our customers. "
    "We use the latest methods and techniques. To this end, we continuously develop our tools and "
    "methods and regularly train our employees. We assure the needed ressources for the projects. "
    "We are responsible to achieve the planned hours for the tasks.")

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