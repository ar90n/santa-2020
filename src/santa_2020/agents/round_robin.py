from .agent import Agent, register

def agent(observation, configuration):
    return observation.step % configuration.banditCount

register(Agent("round_robin", agent))