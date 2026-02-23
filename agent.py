import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["CREWAI_TELEMETRY"] = "false"  # suppress signal handler warning

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

model = LLM("gpt-4o", api_key=openai_api_key)
search_tool = SerperDevTool()

agent = Agent(
    role="Job search assistant",
    goal="Search for job openings in the software engineering field, provide relevant job postings with company name, job title, location, and application link, and summarize key requirements for each.",
    backstory="I am an AI assistant designed to help users find job opportunities in the software engineering field.",
    llm=model,
    tools=[search_tool]
)

job_search_task = Task(
    description="Search for software engineering job openings and provide detailed information about each posting.",
    agent=agent,
    expected_output="A list of 10 relevant job postings with company name, job title, location, application link, and a summary of key requirements for each."
)

job_search_crew = Crew(
    agents=[agent],
    tasks=[job_search_task],
    verbose=True,
    process=Process.sequential
)

result = job_search_crew.kickoff()
st.write(result)