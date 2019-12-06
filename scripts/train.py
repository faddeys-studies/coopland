#!/usr/bin/env python
import logging
import argparse
import threading
import multiprocessing
import os
import dacite
import yaml
import tqdm
import functools
from coopland.game_lib import Direction
from coopland.models.a3c.training import run_training
from coopland.models.a3c import config_lib, reward_lib


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    cli.add_argument("--omp", action="store_true")
    cli.add_argument("--no-threads", action="store_true")
    cli.add_argument("--n-agents", type=int)
    cli.add_argument("--n-games", type=int)
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

    pb = tqdm.tqdm(total=opts.n_games)

    ctx = config_lib.TrainingContext(
        model=cfg.model,
        problem=config_lib.ProblemParams(
            reward_function=reward_lib.reward_function(cfg.reward),
            maze_size=(cfg.maze_size, cfg.maze_size),
            n_agents=opts.n_agents or cfg.model.max_agents,
        ),
        training=cfg.training,
        infrastructure=config_lib.TrainingInfrastructure(
            model_dir=model_dir,
            summaries_dir=model_dir,
            do_visualize=True,
            per_game_callback=functools.partial(per_game_callback, progressbar=pb),
            train_until_n_games=opts.n_games,
        ),
        performance=perf_cfg,
    )
    run_training(ctx)
    pb.close()


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

log = logging.getLogger(__file__)


if __name__ == "__main__":
    main()
