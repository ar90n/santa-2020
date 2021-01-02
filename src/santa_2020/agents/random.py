from .agent import Agent, register

AGENT_KEY = "random"

def agent(obs, conf):
    import random

    return random.randrange(conf.banditCount - 1)


register(AGENT_KEY, lambda resource, comment: Agent(AGENT_KEY, agent, resource, comment))
