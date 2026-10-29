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

st.set_page_config(page_title='Debate and Judge: Cloud Storage Choice', page_icon="🤝", layout="centered")
st.title('Debate and Judge: Cloud Storage Choice')
st.write('Two agents argue for different storage choices, then a judge selects based on the stated needs.')
st.caption("Live LLM demo using LangChain ChatOpenAI. Add a valid OpenAI API key to .env. Agents provide suggestions; verify important decisions.")
request = st.text_area("Your request", value='A small team needs shared files, easy access, and predictable cost.', height=120)
if st.button("Run agents", type="primary"):
    if not request.strip():
        st.warning("Enter a request first.")
    else:
        try:
            context = request.strip()
            st.subheader("Agent discussion")
            for instruction in ['Option A Agent: argue for ease of use; state assumptions.', 'Option B Agent: argue for control and governance; state assumptions.', 'Judge Agent: compare both arguments against the team needs and recommend one.']:
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
