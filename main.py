from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_tavily import TavilySearch
from prompt import PROMPT_TEMPLATE


tools = [TavilySearch()]
llm = ChatGroq(model="llama-3.1-8b-instant")
react_prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

#reasoning engine
agent = create_react_agent(
    llm=llm,
    tools = tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent,
                               tools = tools, 
                               verbose=True)
chain = agent_executor


def main():

    result = chain.invoke(input={"input": "Search for 3 job posting for an ai engineer using langchain in the bay area on LinkedIn and list their details"})
    print(result)


if __name__ == "__main__":
    main()
