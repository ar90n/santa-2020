from .agent import Agent

def agent(observation, configuration):
    return observation.step % configuration.banditCount

round_robin = Agent("round_robin", agent)