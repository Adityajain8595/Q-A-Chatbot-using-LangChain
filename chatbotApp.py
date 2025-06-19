import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initializing chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

## Prompt Template
system_prompt = system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use previous context in chat history to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Answer the question concisely. "
    "\n\n"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

chat_history = []

def generate_response(question, temperature, max_tokens):
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    qa_chain = qa_prompt | llm | StrOutputParser()
    
    response = qa_chain.invoke({"input":question, "chat_history": st.session_state.chat_history}) 

    return response

# Streamlit App

# Sidebar for settings
st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=2048, value=512, step=50)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05)

# Main app
st.title(os.environ['LANGCHAIN_PROJECT'])
st.markdown("<p style='text-align: center;'><b>What do you like to know?<b></p>", unsafe_allow_html=True)
question = st.text_input("Ask any question:")
if question:
    if st.button("Submit"):
        with st.spinner("Generating response..."):
            response = generate_response(question, temperature, max_tokens)
            st.write("Response:", response)

            # Update chat history
            st.session_state.chat_history.extend([
                (HumanMessage(content=question)),
                (AIMessage(content=response))
            ]) 
