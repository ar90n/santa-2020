from santa_2020.agents.random import AGENT_KEY
from .agent import Agent, register

AGENT_KEY = "return_0"

# from https://www.kaggle.com/aatiffraz/a-beginner-s-approach-to-performance-analysis
def agent(obs, conf):
    return 0



register(
    AGENT_KEY, lambda resource, comment: Agent(AGENT_KEY, agent, resource, comment)
)
