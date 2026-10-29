"""Customer Support Case Consensus: Independent agents vote, resolver checks consensus."""
import os
import streamlit as st
from dotenv import load_dotenv  # Load API settings from .env.
from langchain_openai import ChatOpenAI

load_dotenv()  # Read OPENAI_API_KEY from the local .env file.
st.set_page_config(page_title='Customer Support Case Consensus', page_icon="🤝")
st.title('Customer Support Case Consensus')
st.caption('Independent agents vote, resolver checks consensus')
st.markdown('Three agents classify the same support message. The resolver routes only when the votes agree; otherwise it escalates for review.')
case = st.text_area('Customer message', value='I was charged twice for order 78124, and the package shows delivered but I cannot find it.', height=150)  # Edit the fictional case.
model_name = st.text_input("OpenAI model", "gpt-4.1-mini")  # Choose the model used for role prompts.

if st.button("Run workflow", type="primary"):
    api_key = os.getenv("OPENAI_API_KEY")  # Use the value loaded from .env.
    if not api_key:
        st.error("Add OPENAI_API_KEY=your_key to .env, then restart Streamlit.")
    else:
        model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)  # Set up the chat model.
        reports = []  # Keep each agent message for the next handoff.
        def ask_agent(role, task, context):
            prompt = f"You are {role}. {task}\n\nCase facts and prior reports:\n{context}"  # Give each role its task.
            return model.invoke(prompt).content  # Run the role and return its response.
        goal = "Complete the defined task using the available evidence."  # Define the target.
        plan = ask_agent("Workflow planner", "Plan short steps using only these configured roles: Billing classifier, Delivery classifier, Customer care classifier, Consensus resolver", case)  # Ask the LLM to plan.
        st.subheader("Plan")
        st.write(plan)  # Show the plan before actions.
        reports = []  # Store agent messages as observations and handoffs.
        report = ask_agent('Billing classifier', 'Choose Billing, Delivery, or Mixed and give one reason.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Billing classifier', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Delivery classifier', 'Choose Billing, Delivery, or Mixed and give one reason.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Delivery classifier', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Customer care classifier', 'Choose Billing, Delivery, or Mixed and give one reason.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Customer care classifier', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Consensus resolver', 'Compare votes. If categories differ, mark Human review. If they agree, name queue and draft a neutral first response.', case + "\n\nPrior agent messages:\n" + "\n".join(reports))  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Consensus resolver', expanded=True):
            st.write(report)  # Show the agent message and handoff.
        observation = ask_agent("Observer", "Summarize the results and evidence from the agent messages.", case + "\n\nAgent messages:\n" + "\n".join(reports))  # Observe the actions.
        st.subheader("Observed results")
        st.write(observation)
        complete = False
        for attempt in range(2):  # Limit the workflow to two review rounds.
            status = ask_agent("Goal checker", "Reply COMPLETE if goal is met; otherwise NEEDS_REVISION and one specific fix.", goal + "\n\n" + observation)  # Check the result.
            st.write(f"Check {attempt + 1}: {status}")
            if status.strip().upper().startswith("COMPLETE"):
                complete = True
                break  # Stop when the goal checker approves completion.
            observation = ask_agent("Workflow coordinator", "Apply the checker’s fix and return an improved result.", observation + "\n\nChecker: " + status)  # Iterate using feedback.
            st.write(observation)
        if not complete:
            st.warning("Iteration limit reached. A person should review the result.")
        st.info("Fictional classroom simulation. Review before taking action.")
