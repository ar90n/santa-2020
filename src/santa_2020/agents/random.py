from .agent import Agent

def agent(obs, conf):
    import random
    return random.randrange(conf.banditCount - 1)

random = Agent("random", agent)
