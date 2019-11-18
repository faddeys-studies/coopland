import logging
import argparse
import dataclasses
import multiprocessing
import os
import dacite
import yaml
from coopland.game_lib import Direction
from coopland.models.a3c.training import run_training
from coopland.models.a3c import config_lib


@dataclasses.dataclass
class ModelConfig:
    model: config_lib.AgentModelHParams
    training: config_lib.TrainingParams
    maze_size: int


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
    cfg = dacite.from_dict(ModelConfig, cfg)

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

    ctx = config_lib.TrainingContext(
        model=cfg.model,
        problem=config_lib.ProblemParams(
            reward_function=reward_function,
            maze_size=(cfg.maze_size, cfg.maze_size),
            n_agents=opts.n_agents or cfg.model.max_agents,
        ),
        training=cfg.training,
        infrastructure=config_lib.TrainingInfrastructure(
            model_dir=model_dir,
            summaries_dir=model_dir,
            do_visualize=True,
            per_game_callback=per_game_callback,
            train_until_n_games=opts.train_until_n_games
        ),
        performance=perf_cfg,
    )
    run_training(ctx)


def reward_function(maze, replays, exit_pos):
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
            r = 0.5 * (d_old - d_new)
            reward.append(r)
        if replay[-1][2] == exit_pos:
            reward[-1] += 1.0
        rewards.append(reward)
    return rewards


def per_game_callback(replays, immediate_rewards, rewards, critic_values, advantages):
    for replay, immediate_reward, reward, critic_value, advantage in zip(
            replays, immediate_rewards, rewards, critic_values, advantages
    ):
        for (move, _, _), r_i, r_d, v, a in zip(
            replay, immediate_reward, reward, critic_value, advantage
        ):
            for i, d in enumerate(Direction.list_clockwise()):
                dir_msg = f"{d.upper()[:1]}({move.probabilities[i]:.2f}) "
                if i == move.direction_idx:
                    dir_msg = "*" + dir_msg
                move.debug_text += dir_msg
            move.debug_text += (
                f" V={v:.2f}\n" f"Ri={r_i:.2f} " f"Rd={r_d:.2f} " f"A={a:.2f}"
            )


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
