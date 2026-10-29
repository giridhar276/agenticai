import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

def ask_agent(role, task, context, temperature=0.2):
    """Send one role-specific task to the OpenAI model through LangChain."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature)
    prompt = f"You are the {role}.\nTask: {task}\nInput and prior agent messages:\n{context}\n\nGive a concise, practical answer. Separate facts from assumptions. Do not claim to have used tools or verified live information unless the input provides it."
    return llm.invoke(prompt).content

st.set_page_config(page_title='Feedback Loop: Email Improvement Coach', page_icon="🤝", layout="centered")
st.title('Feedback Loop: Email Improvement Coach')
st.write('A drafting agent writes an email, a feedback agent scores it, and a coach recommends a revision.')
st.caption("Live LLM demo using LangChain ChatOpenAI. Add a valid OpenAI API key to .env. Agents provide suggestions; verify important decisions.")
request = st.text_area("Your request", value='Write a polite reminder asking a colleague to share the project file today.', height=120)
if st.button("Run agents", type="primary"):
    if not request.strip():
        st.warning("Enter a request first.")
    else:
        try:
            context = request.strip()
            st.subheader("Agent discussion")
            for instruction in ['Draft Agent: write a first version.', 'Feedback Agent: assess politeness, clarity, and action requested; give specific feedback.', 'Coach Agent: revise the email using the feedback.']:
                role, task = instruction.split(":", 1)
                with st.spinner(f"{role} is working..."):
                    result = ask_agent(role, task.strip(), context)
                st.markdown(f"### {role}")
                st.write(result)
                # Pass each agent's result to the next agent as its working context.
                context += f"\n\n{role} said:\n{result}"
            st.success("Workflow complete. Review the recommendations before acting.")
        except Exception as error:
            st.error("The LLM call did not complete. Check your API key, model access, and internet connection.")
            st.caption(str(error))
st.divider()
st.markdown("**What to observe:** each role has a different task; later agents can use earlier agent messages as context.")
