from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain.messages import HumanMessage
from typing import List

from tools import tavily

load_dotenv()

class ResearchResponse(BaseModel):
    topics: List[str]
    source: List[str]
    summary: str
    tools_used: List[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

system_prompt = """
You are a research agent specialized in gathering and synthesizing information.
Always verify information.
Cite sources.
Be objective.
"""

agent = create_agent(
    model=llm,
    tools=[tavily],
    system_prompt=system_prompt,
    response_format=ResearchResponse
)

response = agent.invoke({
    "messages": [
        HumanMessage(content="Explain quantum computing in simple terms")
    ]
})
print(response)

