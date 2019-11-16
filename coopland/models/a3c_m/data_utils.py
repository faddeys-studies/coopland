import numpy as np
from typing import List, cast
from coopland.game_lib import Observation, Direction


def flip_observations(observations: List[Observation]) -> List[Observation]:
    flipped = []
    for observation in observations:
        agent_id, visibility, corners, visible_other_agents, visible_exit = observation
        flip_visibility = tuple(
            visibility[_dir2i[d.opposite()]] if d in _flip_directions else visibility[i]
            for i, d in enumerate(_directions)
        )
        flip_corners = tuple(
            corners[_dir2i[d.opposite()]][::-1]
            if d in _flip_directions
            else corners[i][::-1]
            for i, d in enumerate(_directions)
        )
        flip_other_agents = [
            (a_id, d.opposite(), dist) if d in _flip_directions else (a_id, d, dist)
            for a_id, d, dist in visible_other_agents
        ]
        if visible_exit[0] is not None:
            ex_d, ex_dist = visible_exit
            flip_visible_exit = (
                ex_d.opposite() if ex_d in _flip_directions else ex_d,
                ex_dist,
            )
        else:
            flip_visible_exit = visible_exit

        flipped.append(
            (
                agent_id,
                flip_visibility,
                flip_corners,
                flip_other_agents,
                flip_visible_exit,
            )
        )

    return cast(List[Observation], flipped)


def flip_direction_id(dir_id):
    d = _directions[dir_id]
    if d in _flip_directions:
        return _dir2i[d.opposite()]
    else:
        return dir_id


def rotate_observations(observations: List[Observation]) -> List[Observation]:
    rotated = []
    for observation in observations:
        agent_id, visibility, corners, visible_other_agents, visible_exit = observation
        rot_visibility = (visibility[-1],) + visibility[:-1]
        rot_corners = (corners[-1],) + corners[:-1]
        rot_other_agents = [
            (a_id, _directions[(_dir2i[d] + 1) % 4], dist)
            for a_id, d, dist in visible_other_agents
        ]
        if visible_exit[0] is not None:
            ex_d, ex_dist = visible_exit
            rot_exit = _directions[(_dir2i[ex_d] + 1) % 4], ex_dist
        else:
            rot_exit = visible_exit
        rotated.append(
            (agent_id, rot_visibility, rot_corners, rot_other_agents, rot_exit)
        )

    return cast(List[Observation], rotated)


def rotate_direction_id(dir_id):
    return (dir_id + 1) % 4


_flip_directions = Direction.East, Direction.West
_directions = Direction.list_clockwise()
_dir2i = {d: i for i, d in enumerate(_directions)}


def get_augmented_training_batch(replay, encode_fn):
    orig_inputs = np.array([move.input_vector for move, _, _ in replay])
    orig_actions = np.array([move.direction_idx for move, _, _ in replay])

    all_inputs = [orig_inputs]
    obs_list = [move.observation for move, _, _ in replay]
    flipped_observations = [flip_observations(obs_list)]
    observations = []
    for _ in range(3):
        obs_list = rotate_observations(obs_list)
        observations.append(obs_list)
        flipped_observations.append(flipped_observations[-1])
    observations += flipped_observations
    all_inputs.extend(np.array(list(map(encode_fn, obs))) for obs in observations)

    all_actions = [orig_actions]
    flipped_actions = [np.array(list(map(flip_direction_id, orig_actions)))]
    for _ in range(3):
        # note: rotate_direction_id can be applied to vectors, so we "cheat" here
        all_actions.append(rotate_direction_id(all_actions[-1]))
        flipped_actions.append(rotate_direction_id(flipped_actions[-1]))
    all_actions += flipped_actions

    all_inputs, all_actions = np.stack(all_inputs), np.stack(all_actions)
    return all_inputs.astype(np.float32), all_actions


def get_training_batch(replay):
    inputs = np.array([move.input_vector for move, _, _ in replay])
    actions = np.array([move.direction_idx for move, _, _ in replay])
    return np.expand_dims(inputs, axis=0), np.expand_dims(actions, axis=0)
