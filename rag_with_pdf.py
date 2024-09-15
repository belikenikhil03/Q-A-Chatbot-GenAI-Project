## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Apply custom background color for the whole app
st.markdown(
    """
    <style>
    .main {
        background-color: #f3e5f5;
    }
    .chat-history-container {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 10px;
    }
    .chat-message {
        background-color: #ffe0b2;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .response-message {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

## Set up Streamlit
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDF's and chat with their content")

## Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

## Check if groq api key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    ## Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    ## Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)
    
    ## Process uploaded PDF's
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            " which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        ## User Input and Chat Display
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            ## Create two sections: Chat History and Assistant Response
            col1, col2 = st.columns([1, 1])  # 50%-50% width split

            with col1:
                st.subheader("Chat History")
                st.markdown("<div class='chat-history-container'>", unsafe_allow_html=True)
                for message in session_history.messages:
                    st.markdown(f"<div class='chat-message'>{message.content}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.subheader("Assistant's Response")
                st.markdown(f"<div class='response-message'>{response['answer']}</div>", unsafe_allow_html=True)

else:
    st.warning("Please enter the GROQ API Key")