from coopland.maze_lib import (
    generate_random_maze,
    flip_maze,
    rotate_maze_clockwise,
    Maze,
    Direction,
)
import pytest


def test_flip_simple_maze():
    maze = Maze(2, 2)
    maze.set_path(0, 0, Direction.East, True)
    maze.set_path(0, 0, Direction.South, True)

    maze_flipped = flip_maze(maze)

    assert maze_flipped.has_path(0, 0, Direction.East)
    assert not maze_flipped.has_path(0, 0, Direction.South)

    assert maze_flipped.has_path(1, 0, Direction.South)
    assert not maze_flipped.has_path(0, 1, Direction.East)


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_flip_maze(width, height):
    maze = generate_random_maze(width, height, 0.1)

    maze_flipped = flip_maze(maze)

    assert maze._horizontal_walls == [
        walls[::-1] for walls in maze_flipped._horizontal_walls
    ]
    assert maze_flipped._vertical_walls == [
        walls[::-1] for walls in maze._vertical_walls
    ]


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_flip_maze_twice(width, height):
    maze = generate_random_maze(width, height, 0.1)

    maze_flipped = flip_maze(flip_maze(maze))

    assert maze._horizontal_walls == maze_flipped._horizontal_walls
    assert maze_flipped._vertical_walls == maze._vertical_walls


def test_rotate_simple_maze():
    maze = Maze(2, 2)
    maze.set_path(0, 0, Direction.East, True)
    maze.set_path(1, 0, Direction.South, True)

    maze_rotated = rotate_maze_clockwise(maze)

    assert not maze_rotated.has_path(0, 0, Direction.East)
    assert not maze_rotated.has_path(0, 0, Direction.South)

    assert maze_rotated.has_path(1, 0, Direction.South)
    assert maze_rotated.has_path(0, 1, Direction.East)


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_rotate_maze_clockwise(width, height):
    maze = generate_random_maze(width, height, 0.1)

    maze_rotated = rotate_maze_clockwise(maze)

    def rotate_2d_list(list2d):
        transposed = [[] for _ in range(len(list2d[0]))]
        for row in list2d[::-1]:
            for i, x in enumerate(row):
                transposed[i].append(x)
        return transposed

    assert maze_rotated._horizontal_walls == rotate_2d_list(maze._vertical_walls)
    assert maze_rotated._vertical_walls == rotate_2d_list(maze._horizontal_walls)


@pytest.mark.parametrize("width,height", [(4, 4), (7, 7), (9, 6), (12, 15)])
def test_rotate_maze_clockwise_4times(width, height):
    maze = generate_random_maze(width, height, 0.1)

    maze_rotated = rotate_maze_clockwise(maze)
    maze_rotated = rotate_maze_clockwise(maze_rotated)
    maze_rotated = rotate_maze_clockwise(maze_rotated)
    maze_rotated = rotate_maze_clockwise(maze_rotated)

    assert maze_rotated._horizontal_walls == maze._horizontal_walls
    assert maze_rotated._vertical_walls == maze._vertical_walls
