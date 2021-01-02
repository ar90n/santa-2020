from .agent import Agent, register

# from https://www.kaggle.com/aatiffraz/a-beginner-s-approach-to-performance-analysis
def agent(obs, conf):
    return 0


register(Agent("return_0", agent))