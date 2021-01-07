from santa_2020.agents.random import AGENT_KEY
from .agent import Agent, register

AGENT_KEY = "round_robin"


def agent(observation, configuration):
    return observation.step % configuration.banditCount


register(
    AGENT_KEY, lambda resource, comment, name: Agent(AGENT_KEY, agent, resource, comment, name)
)
