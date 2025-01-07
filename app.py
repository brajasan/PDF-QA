import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from tempfile import NamedTemporaryFile
import time

# Streamlit app setup
st.set_page_config(page_title="PDF QA")

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""You are a knowledgeable chatbot, here to help with questions of the user. 
Your tone should be casual and informative.\n\nContext: {context}\n\nUser: {question}\nChatbot:"""
        )
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'llm' not in st.session_state:
        st.session_state.llm = Ollama(
            base_url="http://localhost:11434", 
            model="llama3.2:1b", 
            verbose=True, 
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

# Function to create a vectorstore
def create_vectorstore(pdf_path):
    if not os.path.exists("files"):
        os.makedirs("files")  # Ensure the directory exists
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model="llama3.2:1b"))
    return vectorstore

# Initialize session state
initialize_session_state()

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            # Use temporary file to handle uploaded data
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Create vectorstore
            st.session_state.vectorstore = create_vectorstore(tmp_file_path)
            retriever = st.session_state.vectorstore.as_retriever()

            # Create QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=retriever,
                verbose=True,
                chain_type_kwargs={"prompt": st.session_state.prompt},
            )
            st.success("PDF processed successfully!")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Chat interface
if st.session_state.qa_chain:
    user_input = st.text_input("You:", key="user_input")
    
    if user_input:
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                try:
                    response = st.session_state.qa_chain(user_input)
                    full_response = response['result']
                    st.markdown(full_response)
                    
                    chatbot_message = {"role": "assistant", "message": full_response}
                    st.session_state.chat_history.append(chatbot_message)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
