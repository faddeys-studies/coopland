import random
from coopland.maze_lib import Direction


_directions = Direction.list_clockwise()


class RandomMove:
    def __init__(self):
        self.direction = random.choice(_directions)

    def __repr__(self):
        return f"<{self.direction}>"


def agent_fn(agent_id, visibility, visible_other_agents, visible_exit):
    del agent_id
    del visibility
    del visible_other_agents
    del visible_exit
    return RandomMove()


agent_fn.name = "random"
