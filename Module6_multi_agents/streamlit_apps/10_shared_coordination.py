import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather and today's forecast for a city using Open-Meteo."""
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=15).json()
    places = geo.get("results", [])
    if not places:
        return f"No location found for {city}."
    place = places[0]
    weather = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": place["latitude"], "longitude": place["longitude"], "current": "temperature_2m,precipitation,wind_speed_10m", "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "timezone": "auto", "forecast_days": 1}, timeout=15).json()
    return f"Source: Open-Meteo. Location: {place['name']}, {place.get('country','')}. Data: {weather}"

@tool
def convert_currency(amount: float, source_currency: str, target_currency: str) -> str:
    """Convert an amount using the latest available reference exchange rate."""
    data = requests.get(f"https://api.frankfurter.dev/v1/latest", params={"base": source_currency.upper(), "symbols": target_currency.upper()}, timeout=15).json()
    rate = data.get("rates", {}).get(target_currency.upper())
    if rate is None:
        return f"No exchange rate returned for {source_currency} to {target_currency}."
    return f"Source: Frankfurter. Rate date: {data.get('date')}. {amount:.2f} {source_currency.upper()} = {amount*rate:.2f} {target_currency.upper()} (rate {rate})."

def call_tool(tool_function, **inputs):
    """Run a real API-backed LangChain tool and return its result."""
    return tool_function.invoke(inputs)

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

def ask_agent(role, task, context, temperature=0.2):
    """Send one role-specific task to the OpenAI model through LangChain."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=temperature)
    prompt = f"You are the {role}.\nTask: {task}\nInput and prior agent messages:\n{context}\n\nGive a concise, practical answer. Separate facts from assumptions. Do not claim to have used tools or verified live information unless the input provides it."
    return llm.invoke(prompt).content

st.set_page_config(page_title='Shared Coordination: Conference Preparation', page_icon="🤝", layout="centered")
st.title('Shared Coordination: Conference Preparation')
st.write('Travel, session, and checklist agents coordinate around one event and update a shared plan.')
st.caption("Live LLM and external API tool demo using LangChain ChatOpenAI. Add a valid OpenAI API key to .env. Agents use live external data and provide suggestions; verify important decisions.")
city = st.text_input("City for live weather", value="Bengaluru" if "conference" in '10_shared_coordination' else "Hyderabad")
request = st.text_area("Your request", value='Attend a one-day conference in Bengaluru next Friday.', height=120)
if st.button("Run agents", type="primary"):
    if not request.strip():
        st.warning("Enter a request first.")
    else:
        try:
            context = request.strip()
            with st.spinner("Calling the live Open-Meteo weather tool..."):
                live_tool_result = call_tool(get_weather, city=city)
            st.markdown("### External tool: Open-Meteo weather")
            st.write(live_tool_result)
            context += "\n\nExternal tool result (Open-Meteo): " + str(live_tool_result)
            st.subheader("Agent discussion")
            for instruction in ['Travel Agent: list arrival and local transport items to confirm.', 'Session Agent: list agenda and session items to check.', 'Coordinator Agent: merge reports into a single checklist and flag missing details.']:
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
