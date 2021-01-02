from .agent import Agent, register


def agent(obs, conf):
    import random

    return random.randrange(conf.banditCount - 1)


register(Agent("random", agent))
