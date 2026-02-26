import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.makedirs("output", exist_ok=True)

# Disable CrewAI telemetry BEFORE importing anything from crewai
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

st.title("Stock Picker Agent")
st.write("AI-powered investment research using 3 specialized agents")

sector = st.selectbox("Choose a sector:", ["technology", "healthcare", "energy", "finance"])

if st.button("Run Analysis"):
    with st.spinner("Agents are researching... (this takes 2-5 minutes)"):
        # Import INSIDE the button click so it runs after Streamlit is ready
        from stock_picker import stock_picker_crew
        result = stock_picker_crew.kickoff(inputs={"sector": sector})
    st.markdown(result.raw)
    st.download_button("Download Report", result.raw, file_name="stock_recommendations.md")
