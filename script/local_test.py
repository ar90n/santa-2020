from numpy.lib.npyio import loads
from santa_2020 import simulate, agents, io, util
import sys
from pathlib import Path

target_agents = [
    agents.construct(**io.load_agent_conf(Path(path))) for path in sys.argv[1:]
]
#enemy_agents = [agents.construct("random"), agents.construct("round_robin")]
enemy_agents = []

ret = simulate.run(target_agents, enemy_agents).unwrap()
for k, r in ret.items():
    print("=" * 10)
    print(r.duration)
    print(f"{k[0]}:{r.result[-1][0]['reward']}")
    print(f"{k[1]}:{r.result[-1][1]['reward']}")
