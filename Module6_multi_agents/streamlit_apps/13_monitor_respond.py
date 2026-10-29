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

st.set_page_config(page_title='Monitor and Respond: Service Alert', page_icon="🤝", layout="centered")
st.title('Monitor and Respond: Service Alert')
st.write('A monitoring agent spots an alert, an investigator checks sample signals, and a responder proposes a safe action.')
st.caption("Live LLM demo using LangChain ChatOpenAI. Add a valid OpenAI API key to .env. Agents provide suggestions; verify important decisions.")
request = st.text_area("Your request", value='Checkout error rate increased to 8% in the last 10 minutes.', height=120)
if st.button("Run agents", type="primary"):
    if not request.strip():
        st.warning("Enter a request first.")
    else:
        try:
            context = request.strip()
            st.subheader("Agent discussion")
            for instruction in ['Monitor Agent: summarize the alert and what needs checking.', 'Investigator Agent: reason from provided signals; clearly separate facts from hypotheses.', 'Responder Agent: suggest safe investigation and communication steps; do not change production systems.']:
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
