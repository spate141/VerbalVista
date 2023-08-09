from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


class GoogleSerper:

    def __init__(self):
        pass

    @staticmethod
    def load_google_serper_agent(temperature: float = 0.0, verbose: bool = True):
        llm = OpenAI(temperature=temperature)
        tools = load_tools(["google-serper"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
        return agent

