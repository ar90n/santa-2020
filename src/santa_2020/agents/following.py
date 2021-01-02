from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "following"


@context
def agent(stats: BanditStats, env: Environment) -> int:
    if stats.step == 0:
        return stats.step
    else:
        return stats.last_opp_action

register(
    AGENT_KEY, lambda resource, comment: Agent(AGENT_KEY, agent, resource, comment)
)

