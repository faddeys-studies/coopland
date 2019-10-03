import random
from coopland.maze_lib import Direction, Maze


class Game:
    def __init__(self, maze: Maze, agent_fn, n_agents):
        self.maze = maze
        self.agent_fn = agent_fn

        all_points = [(x, y) for x in range(maze.width) for y in range(maze.height)]
        positions = random.sample(all_points, n_agents + 1)
        self.agent_positions = positions[1:]
        self.exit_position = positions[0]

        self.directions = Direction.list_clockwise()
        dir2i = {d: i for i, d in enumerate(self.directions)}
        self.observations = [
            [[0, 0, 0, 0] for _ in range(maze.height)] for _ in range(maze.width)
        ]
        for y in range(maze.height):
            n = 0
            for x in range(1, maze.width+1):
                if maze.has_path(x - 1, y, Direction.East):
                    n += 1
                    self.observations[x][y][dir2i[Direction.West]] = n
                else:
                    for dx in range(n+1):
                        self.observations[x - dx - 1][y][dir2i[Direction.East]] = dx
                    n = 0

        for x in range(maze.width):
            n = 0
            for y in range(1, maze.height+1):
                if maze.has_path(x, y - 1, Direction.South):
                    n += 1
                    self.observations[x][y][dir2i[Direction.North]] = n
                else:
                    for dy in range(n+1):
                        self.observations[x][y - dy - 1][dir2i[Direction.South]] = dy
                    n = 0


