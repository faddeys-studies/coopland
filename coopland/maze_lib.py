import enum
import random
import time


class Direction(str, enum.Enum):
    North = 'north'
    South = 'south'
    East = 'east'
    West = 'west'

    def apply(self, x, y, d=1):
        if self == self.North:
            return x, y-d
        if self == self.South:
            return x, y+d
        if self == self.East:
            return x+d, y
        if self == self.West:
            return x-d, y
        raise ValueError(self)

    @classmethod
    def list_clockwise(cls):
        return [cls.North, cls.East, cls.South, cls.West]

    def opposite(self):
        if self == self.North:
            return self.South
        if self == self.South:
            return self.North
        if self == self.East:
            return self.West
        if self == self.West:
            return self.East
        raise ValueError(self)


class Maze:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.generation_seed = None
        self._vertical_walls = [[True] * (width-1) for _ in range(height)]
        self._horizontal_walls = [[True] * width for _ in range(height-1)]

    def has_path(self, x, y, direction):
        direction = Direction(direction)
        if direction == Direction.East:
            return x < self.width-1 and not self._vertical_walls[y][x]
        elif direction == Direction.West:
            return x > 0 and not self._vertical_walls[y][x-1]
        elif direction == Direction.North:
            return y > 0 and not self._horizontal_walls[y-1][x]
        else:
            return y < self.height-1 and not self._horizontal_walls[y][x]

    def set_path(self, x, y, direction, has_path):
        direction = Direction(direction)
        has_wall = not has_path
        if direction == Direction.East:
            self._vertical_walls[y][x] = has_wall
        elif direction == Direction.West:
            self._vertical_walls[y][x-1] = has_wall
        elif direction == Direction.North:
            self._horizontal_walls[y-1][x] = has_wall
        else:
            self._horizontal_walls[y][x] = has_wall

    def is_in(self, x, y):
        return (0 <= x < self.width) and (0 <= y < self.height)


def generate_random(width, height, branching, seed=None):
    if seed is None:
        seed = int(time.time_ns())
    rnd = random.Random()
    rnd.seed(seed)

    heads = [(rnd.randrange(width), rnd.randrange(height))]

    visited = {heads[0]}
    n_not_visited = width * height - 1

    maze = Maze(width, height)
    maze.generation_seed = seed
    directions = [*Direction]

    while n_not_visited > 0:
        assert len(heads) > 0
        new_heads = []
        for x, y in heads:
            rnd.shuffle(directions)
            for d in directions:
                x1, y1 = d.apply(x, y)
                p1 = (x1, y1)
                if not maze.is_in(x1, y1):
                    continue
                if p1 in visited:
                    continue

                maze.set_path(x, y, d, True)
                visited.add(p1)
                n_not_visited -= 1
                new_heads.append(p1)

                if rnd.random() > branching:
                    break

        if not new_heads:
            candidates = []
            for x, y in visited:
                for d in directions:
                    p1 = d.apply(x, y)
                    if p1 not in visited:
                        candidates.append((x, y))
                        break
            if candidates:
                new_heads.append(rnd.choice(candidates))

        heads = new_heads

    return maze
