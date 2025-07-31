import os
import tempfile
import streamlit as st

# LangChain imports for building a conversational retrieval-augmented generation (RAG) pipeline:
# - `create_history_aware_retriever`: enhances the retriever by considering chat history for more context-aware results.
# - `create_retrieval_chain`: sets up the core retrieval chain that integrates the retriever and document combiner.
# - `create_stuff_documents_chain`: used to combine retrieved documents into a prompt for the language model using the "stuff" strategy.
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_chroma import Chroma # vector store DB
from langchain.vectorstores import FAISS

# Imports for managing chat history in LangChain:
# - `ChatMessageHistory` (from langchain_community): provides a concrete implementation to store and manage past chat messages.
# - `BaseChatMessageHistory` (from langchain_core): defines the abstract base interface for chat history handling, useful for custom implementations.
# `RunnableWithMessageHistory` wraps a LangChain Runnable (e.g., retrieval or conversation chain) 
# to enable memory integration for contextual conversations.
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## Set up streamlit app
st.title("Coversational RAG with PDF uploads and chat history")
st.write("Upload PDF and chat with their content")

api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")



if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")
    session_id = st.sidebar.text_input("Session ID", value="default-session")

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    upload_files = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if upload_files:
        documents = []
        for upload_file in upload_files:
            # temppdf = f"./temp.pdf"
            # with open(temppdf, "wb") as file:
            #     file.write(upload_file.getvalue())
            #     file_name = upload_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(upload_file.getvalue())
                file_path = tmp_file.name

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            os.unlink(file_path)  #delete the temp file now that it's no longer needed
        st.write(f"Total documents: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # vectorStore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        vectorStore = FAISS.from_documents(documents=chunks, embedding=embeddings)

        # res = vectorStore.similarity_search("WHat is my name ?")
        # st.write(res[0].page_content)

        reriever = vectorStore.as_retriever()

        chat_aware_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and intelligent assistant. Use the previous conversation to understand what the user wants and answer the latest question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        chat_aware_history = create_history_aware_retriever(llm,reriever,prompt=chat_aware_qa_prompt)

        system_prompt = (
            "You are an assistent for question-answering tasks."
            "Use the following peaces of retrieved context to answe the question"
            "the question you don't know the answer say that you don't know"
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(chat_aware_history, document_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        chat_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        user_input = st.text_input("Type your question...")
        if user_input:
            session_history = get_session_history(session_id)
            response = chat_with_history.invoke(
                {"input": user_input},
                config = {
                    "configurable": {"session_id": session_id}
                }
            )
            final_output = {
                "output": response["answer"],  # what LangChain tracing expects
                **response                     # keep rest of the keys like input/context
            }
            st.write(final_output)
            
            st.write(final_output["output"])
            if "answer" not in response:
                st.error("❌ 'answer' key missing in chain response!")
else:
    st.warning("Please enter an API key!!")

    