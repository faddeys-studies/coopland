import numpy as np
import itertools
from coopland.models.a3c import config_lib
from coopland.maze_lib import Direction


def reward_function(params: config_lib.RewardParams):
    def compute_reward(maze, replays, exit_pos):
        distances = {}

        _q = [(exit_pos, 0)]
        while _q:
            pos, dist = _q.pop(0)
            if pos not in distances or dist < distances[pos]:
                distances[pos] = dist
                for d in _directions:
                    if maze.has_path(*pos, direction=d):
                        next_pos = d.apply(*pos)
                        if next_pos not in distances or dist + 1 < distances[next_pos]:
                            _q.append((next_pos, dist + 1))

        rewards = []
        for replay in replays:
            reward = []
            for t, (move, old_pos, new_pos) in enumerate(replay):
                d_old = distances[old_pos]
                d_new = distances[new_pos]
                r = params.step_reward * (d_old - d_new)
                reward.append(r)
            rewards.append(reward)
        if params.average_over_team:
            if len(rewards) > 1:
                average_reward = []
                for rew_tup in itertools.zip_longest(*rewards):
                    rew_tup = [r for r in rew_tup if r is not None]
                    average_reward.append(sum(rew_tup) / len(rew_tup))
            else:
                average_reward = rewards[0]
            average_reward = np.array(average_reward)
            n_left = sum(replay[-1][2] == exit_pos for replay in replays)
            if n_left == len(replays):
                average_reward[-1] += params.exit_reward
            average_reward[-1] += params.one_agent_exit_reward * n_left
            average_reward = discount(average_reward, params.discount_rate)
            rewards = [average_reward[: len(replay)] for replay in replays]
        else:
            for reward, replay in zip(rewards, replays):
                if replay[-1][2] == exit_pos:
                    reward[-1] += params.exit_reward
            rewards = [
                discount(np.array(reward), params.discount_rate) for reward in rewards
            ]
        if params.failure_reward is not None:
            for i, replay in enumerate(replays):
                if replay[-1][2] != exit_pos:
                    rewards[i][:] += params.failure_reward
        if params.overall_shift is not None:
            for i, replay in enumerate(replays):
                rewards[i][:] += params.overall_shift
        return rewards

    return compute_reward


def discount(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    next_r = discounted_rewards[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        r = rewards[i]
        next_r = gamma * next_r + r
        discounted_rewards[i] = next_r
    return discounted_rewards


def get_visible_positions(visibility, position):
    result = {position}
    for d, v in zip(_directions, visibility):
        for dist in range(1, v + 1):
            result.add(d.apply(*position, dist))
    return result


_directions = Direction.list_clockwise()
