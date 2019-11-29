import logging
import argparse
import threading
import multiprocessing
import os
import dacite
import yaml
import tqdm
import functools
import itertools
import numpy as np
from coopland.game_lib import Direction
from coopland.a3c.training import run_training
from coopland import config_lib


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    cli.add_argument("--omp", action="store_true")
    cli.add_argument("--no-threads", action="store_true")
    cli.add_argument("--n-agents", type=int)
    cli.add_argument("--train-until-n-games", type=int)
    opts = cli.parse_args()
    logging.basicConfig(level=logging.INFO)

    model_dir = opts.model_dir

    with open(os.path.join(model_dir, "config.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg = dacite.from_dict(config_lib.ModelConfig, cfg)

    if opts.no_threads:
        perf_cfg = config_lib.PerformanceParams(
            system_supports_omp=False,
            omp_thread_limit=0,
            multithreaded_training=False,
            session_config=None,
        )
    elif opts.omp:
        perf_cfg = config_lib.PerformanceParams(
            system_supports_omp=True,
            omp_thread_limit=multiprocessing.cpu_count() // 2,
            multithreaded_training=True,
            session_config=None,
        )
    else:
        perf_cfg = config_lib.PerformanceParams(
            system_supports_omp=False,
            omp_thread_limit=0,
            multithreaded_training=True,
            session_config=None,
        )

    pb = tqdm.tqdm(total=opts.train_until_n_games)

    default_n_agents = getattr(cfg.model_hparams, "max_agents", 1)
    ctx = config_lib.TrainingContext(
        model_type=cfg.model_type,
        model_hparams=cfg.model_hparams,
        problem=config_lib.ProblemParams(
            reward_function=reward_function(cfg.reward),
            maze_size=(cfg.maze_size, cfg.maze_size),
            n_agents=opts.n_agents or default_n_agents,
        ),
        training=cfg.training,
        infrastructure=config_lib.TrainingInfrastructure(
            model_dir=model_dir,
            summaries_dir=model_dir,
            do_visualize=True,
            per_game_callback=functools.partial(per_game_callback, progressbar=pb),
            train_until_n_games=opts.train_until_n_games,
        ),
        performance=perf_cfg,
    )
    run_training(ctx)
    pb.close()


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
        if len(rewards) > 1:
            average_reward = []
            for rew_tup in itertools.zip_longest(*rewards):
                rew_tup = [r for r in rew_tup if r is not None]
                average_reward.append(sum(rew_tup) / len(rew_tup))
        else:
            average_reward = rewards[0]
        average_reward = np.array(average_reward)
        if all(replay[-1][2] == exit_pos for replay in replays):
            average_reward[-1] += params.exit_reward
        average_reward = discount(average_reward, params.discount_rate)
        rewards = [average_reward[:len(replay)] for replay in replays]
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


def per_game_callback(
    worker_id,
    game_index,
    replays,
    rewards,
    critic_values,
    advantages,
    progressbar: tqdm.tqdm,
):
    del worker_id
    for replay, reward, critic_value, advantage in zip(
        replays, rewards, critic_values, advantages
    ):
        for (move, _, _), r, v, a in zip(replay, reward, critic_value, advantage):
            for i, d in enumerate(Direction.list_clockwise()):
                dir_msg = f"{d.upper()[:1]}({move.probabilities[i]:.2f}) "
                if i == move.direction_idx:
                    dir_msg = "*" + dir_msg
                move.debug_text += dir_msg
            move.debug_text += f" V={v:.2f} R={r:.2f} A={a:.2f}"
    with _pb_lock:
        progressbar.n = max(game_index, progressbar.n)
        progressbar.update(0)


_pb_lock = threading.Lock()


def get_visible_positions(visibility, position):
    result = {position}
    for d, v in zip(_directions, visibility):
        for dist in range(1, v + 1):
            result.add(d.apply(*position, dist))
    return result


_directions = Direction.list_clockwise()
log = logging.getLogger("coopland.models.a3c.main")


if __name__ == "__main__":
    main()
