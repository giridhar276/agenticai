"""IT Incident Response Debate: Agents discuss and challenge each other."""
import os
import streamlit as st
from dotenv import load_dotenv  # Load API settings from .env.
from langchain_openai import ChatOpenAI

load_dotenv()  # Read OPENAI_API_KEY from the local .env file.
st.set_page_config(page_title='IT Incident Response Debate', page_icon="🤝")
st.title('IT Incident Response Debate')
st.caption('Agents discuss and challenge each other')
st.markdown('Application, database, and network agents offer competing hypotheses about a fictional checkout outage. An incident lead separates evidence from assumptions.')
case = st.text_area('Incident details', value='10:05 AM checkout response time rose from 1 to 12 seconds. CPU is normal. Database connection pool is at 98%. Deployment occurred at 9:50 AM. Some users see timeouts.', height=150)  # Edit the fictional case.
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
        plan = ask_agent("Workflow planner", "Plan short steps using only these configured roles: Application agent, Database agent, Network agent, Incident lead", case)  # Ask the LLM to plan.
        st.subheader("Plan")
        st.write(plan)  # Show the plan before actions.
        reports = []  # Store agent messages as observations and handoffs.
        report = ask_agent('Application agent', 'Give one likely cause and evidence from the facts.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Application agent', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Database agent', 'Give one likely cause and evidence from the facts.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Database agent', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Network agent', 'Give one likely cause and evidence from the facts.', case)  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Network agent', expanded=False):
            st.write(report)  # Show the agent message and handoff.
        report = ask_agent('Incident lead', 'Compare views, point out disagreement, separate facts from hypotheses, and list next checks. Do not claim a confirmed cause.', case + "\n\nPrior agent messages:\n" + "\n".join(reports))  # Run this agent role.
        reports.append(report)  # Keep its message for later roles.
        with st.expander('Incident lead', expanded=True):
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
