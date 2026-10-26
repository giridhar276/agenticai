# Run: python -m streamlit run 01_chatbot.py
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError

load_dotenv(Path(__file__).parent / '.env')

def get_llm():
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Add OPENAI_API_KEY to the .env file, then restart Streamlit.')
        st.stop()
    return ChatOpenAI(model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
                      temperature=0, timeout=45, max_retries=2)

def show_error(exc):
    # Avoid displaying raw exceptions, which may contain submitted data.
    if isinstance(exc, AuthenticationError):
        st.error('Authentication failed. Check your OpenAI API key.')
    elif isinstance(exc, RateLimitError):
        st.error('API rate limit or quota reached. Check billing and try again later.')
    elif isinstance(exc, (APIConnectionError, APITimeoutError)):
        st.error('Could not reach the API. Check your network and try again.')
    else:
        st.error('The request failed or returned an invalid result. Check model access and try again.')

def intro(title):
    st.set_page_config(page_title=title, layout='wide')
    st.title(title)
    st.caption('Python 3.11.9 • LangChain OpenAI • API calls happen only when you submit.')
    st.info('Submitted text is sent to the OpenAI API. Use synthetic data for demonstrations.')

import streamlit as st

intro('01 — Simple LLM Chatbot')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if st.button('Clear conversation'):
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

if question := st.chat_input('Ask a question about Generative AI'):
    with st.chat_message('user'):
        st.write(question)
    history = st.session_state.messages[-10:]
    try:
        with st.chat_message('assistant'), st.spinner('Thinking...'):
            llm = get_llm()
            response = llm.invoke([
                ('system', 'You are a helpful trainer. Explain clearly and concisely.'),
                *[(m['role'], m['content']) for m in history],
                ('user', question),
            ])
            st.write(response.content)
        st.session_state.messages.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response.content},
        ])
    except Exception as exc:
        show_error(exc)
