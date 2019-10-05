import random
import types
from typing import List, Tuple, Optional
from coopland.maze_lib import Direction, Maze


VisibilityItem = Tuple[int, int, int, int]
Position = Tuple[int, int]


class Move:
    direction: Direction


ReplayItem = Tuple[Optional[Move], Position, Position]
Replay = List[ReplayItem]
AllAgentReplays = List[Replay]


class Game:
    def __init__(self, maze: Maze, agent_fn, n_agents):
        self.maze = maze
        self.agent_fn = agent_fn

        all_points = [(x, y) for x in range(maze.width) for y in range(maze.height)]
        rnd = random.Random()
        if maze.generation_seed is not None:
            rnd.seed(maze.generation_seed)
        positions = rnd.sample(all_points, n_agents + 1)
        self.initial_agent_positions = positions[1:]
        self.n_agents = n_agents
        self.exit_position = positions[0]

        self.directions = Direction.list_clockwise()
        self._dir2i = dir2i = {d: i for i, d in enumerate(self.directions)}
        self.visibility = [
            [[0, 0, 0, 0] for _ in range(maze.height)] for _ in range(maze.width)
        ]
        self.corners = [
            [[[0, 0], [0, 0], [0, 0], [0, 0]] for _ in range(maze.height)]
            for _ in range(maze.width)
        ]
        for y in range(maze.height):
            n = 0
            for x in range(1, maze.width + 1):
                if maze.has_path(x - 1, y, Direction.East):
                    n += 1
                    self.visibility[x][y][dir2i[Direction.West]] = n
                else:
                    for dx in range(n + 1):
                        self.visibility[x - dx - 1][y][dir2i[Direction.East]] = dx
                    n = 0

        for x in range(maze.width):
            n = 0
            for y in range(1, maze.height + 1):
                if maze.has_path(x, y - 1, Direction.South):
                    n += 1
                    self.visibility[x][y][dir2i[Direction.North]] = n
                else:
                    for dy in range(n + 1):
                        self.visibility[x][y - dy - 1][dir2i[Direction.South]] = dy
                    n = 0

        for d_i, d in enumerate(self.directions):
            d_left_i = (d_i - 1) % 4
            d_right_i = (d_i + 1) % 4
            for x in range(maze.width):
                for y in range(maze.height):
                    vis = self.visibility[x][y]
                    corners = self.corners[x][y]
                    for dist in range(1, vis[d_i] + 1):
                        x1, y1 = d.apply(x, y, dist)
                        vis1 = self.visibility[x1][y1]
                        if vis1[d_left_i] > 0:
                            corners[d_i][0] += 1
                        if vis1[d_right_i] > 0:
                            corners[d_i][1] += 1

    def play(self, max_steps) -> AllAgentReplays:
        replays: AllAgentReplays = [[] for _ in self.initial_agent_positions]
        agent_positions = self.initial_agent_positions[:]

        for t in range(max_steps):
            moves = []
            for a, p in enumerate(agent_positions):
                if p == self.exit_position:
                    moves.append(None)
                else:
                    visible_other_agents = []
                    for aa, ap in enumerate(agent_positions):
                        if a == aa:
                            continue
                        d, dist = self._get_visibility(p, ap)
                        if d is not None:
                            visible_other_agents.append((aa, d, dist))
                    visible_exit = self._get_visibility(p, self.exit_position)
                    visibility = self.visibility[p[0]][p[1]]
                    corners = self.corners[p[0]][p[1]]
                    moves.append(
                        self.agent_fn(
                            a, visibility, corners, visible_other_agents, visible_exit
                        )
                    )
            if not any(moves):
                break
            for i, move in enumerate(moves):
                if move is None:
                    continue
                d = move.direction
                p_from = p_to = agent_positions[i]
                if d is not None:
                    if self.maze.has_path(*p_from, d):
                        p_to = d.apply(*p_from)
                        agent_positions[i] = p_to
                replays[i].append((move, p_from, p_to))

        return replays

    def _get_visibility(self, p1, p2):
        if p1 == p2:
            return Direction.North, 0
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            d = Direction.North if y1 > y2 else Direction.South
            dist = abs(y2 - y1)
            if self.visibility[x1][y1][self._dir2i[d]] >= dist:
                return d, dist
        if y1 == y2:
            d = Direction.West if x1 > x2 else Direction.East
            dist = abs(x2 - x1)
            if self.visibility[x1][y1][self._dir2i[d]] >= dist:
                return d, dist
        return None, None

    def get_agent_name(self):
        if hasattr(self.agent_fn, "name"):
            return self.agent_fn.name
        if isinstance(self.agent_fn, types.FunctionType):
            return f"{self.agent_fn.__module__}.{self.agent_fn.__name__}"
        return repr(self.agent_fn)
