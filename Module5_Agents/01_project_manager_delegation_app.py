"""Project Manager Delegates to Child Agents: Master agent delegates to child agents."""
import os
import streamlit as st
from dotenv import load_dotenv  # Load API settings from .env.
from langchain_openai import ChatOpenAI

load_dotenv()  # Read OPENAI_API_KEY from the local .env file.
st.set_page_config(page_title='Project Manager Delegates to Child Agents', page_icon="🤝")
st.title('Project Manager Delegates to Child Agents')
st.caption('Master agent delegates to child agents')
st.markdown('A delivery manager coordinates a delayed software launch. Specialist agents assess schedule, quality, and customer communications.')
case = st.text_area('Project situation', value='Payment feature launch delayed by 5 days. Testing found 3 medium defects. Customer launch is in 2 weeks.', height=150)  # Edit the fictional case.
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
        plan = ask_agent("Workflow planner", "Plan short steps using only these configured roles: Schedule specialist, Quality specialist, Customer communications specialist, Project manager", case)  # Ask the LLM to plan.
        st.subheader("Plan")
        st.write(plan)  # Show the plan before actions.
        reports = []  # Store agent messages as observations and handoffs.
        report = ask_agent('Schedule specialist', 'Assess schedule impact and propose a recovery action.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Schedule specialist', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Quality specialist', 'Assess testing risk and recommend release checks.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Quality specialist', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Customer communications specialist', 'Draft a factual customer update. Do not promise a date.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Customer communications specialist', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Project manager', 'Combine the specialist reports into an action plan. Identify dependencies and decisions a human must approve.', case + "\n\nPrior agent messages:\n" + "\n".join(reports))  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Project manager', expanded=True):
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
