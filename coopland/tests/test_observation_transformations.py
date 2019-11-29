from coopland.game_lib import Game, Move, Direction
from coopland.maze_lib import generate_random_maze, flip_maze, rotate_maze_clockwise
from coopland.a3c import data_utils
import pytest
import random


class DummyAgent:
    _directions = Direction.list_clockwise()

    def __init__(self, moves_to_do=None):
        self.observations = []
        self.move_dir_ids = []
        self.moves_to_do = moves_to_do
        self.t = 0

    def __call__(self, *observation):
        self.observations.append(observation)
        if self.moves_to_do is None:
            visibility = observation[1]
            dir_idx = random.choices(range(len(visibility)), weights=visibility)[0]
        else:
            dir_idx = self.moves_to_do[self.t]
        self.t += 1
        self.move_dir_ids.append(dir_idx)
        move = Move()
        move.direction = self._directions[dir_idx]
        return move


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_flip_observations(width, height):
    maze = generate_random_maze(width, height, 0.1)
    agent = DummyAgent()
    Game(maze, agent, [(width // 2, height // 2)], (0, 0)).play(30)

    maze_flipped = flip_maze(maze)
    agent2 = DummyAgent(list(map(data_utils.flip_direction_id, agent.move_dir_ids)))
    Game(
        maze_flipped, agent2, [(width - width // 2 - 1, height // 2)], (width - 1, 0)
    ).play(30)

    assert agent2.observations == data_utils.flip_observations(agent.observations)


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_flip_observations_twice(width, height):
    maze = generate_random_maze(width, height, 0.1)
    agent = DummyAgent()
    Game(maze, agent, [(width // 2, height // 2)], (0, 0)).play(30)

    assert agent.move_dir_ids == list(
        map(
            data_utils.flip_direction_id,
            map(data_utils.flip_direction_id, agent.move_dir_ids),
        )
    )
    assert agent.observations == data_utils.flip_observations(
        data_utils.flip_observations(agent.observations)
    )


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_rotate_observations(width, height):
    maze = generate_random_maze(width, height, 0.1)
    agent = DummyAgent()
    Game(maze, agent, [(width // 2, height // 2)], (0, 0)).play(30)

    maze_rotated = rotate_maze_clockwise(maze)
    agent2 = DummyAgent(list(map(data_utils.rotate_direction_id, agent.move_dir_ids)))
    Game(
        maze_rotated, agent2, [(height - height // 2 - 1, width // 2)], (width - 1, 0)
    ).play(30)

    assert agent2.observations == data_utils.rotate_observations(agent.observations)


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_rotate_observations_4times(width, height):
    maze = generate_random_maze(width, height, 0.1)
    agent = DummyAgent()
    Game(maze, agent, [(width // 2, height // 2)], (0, 0)).play(30)

    rotated_dir_ids = agent.move_dir_ids
    rotated_observations = agent.observations
    for _ in range(4):
        rotated_dir_ids = list(map(data_utils.rotate_direction_id, rotated_dir_ids))
        rotated_observations = data_utils.rotate_observations(rotated_observations)

    assert agent.move_dir_ids == rotated_dir_ids
    assert agent.observations == rotated_observations


# TODO add tests for multi-agent case (to test transform of visible other agents)
