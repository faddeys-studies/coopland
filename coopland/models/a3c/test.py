import argparse
import os
import yaml
import dacite
import tqdm
import numpy as np
import tensorflow as tf
from coopland.models.a3c import agent, config_lib
from coopland import game_lib, maze_lib


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    cli.add_argument("--size", type=int)
    cli.add_argument("--n-agents", type=int)
    cli.add_argument("--n-games", type=int, default=10000)
    cli.add_argument("--max-game-len", type=int, default=100)
    cli.add_argument("--out")
    opts = cli.parse_args()

    model_dir = opts.model_dir
    if opts.out is None:
        opts.out = os.path.join(model_dir, "test-results.txt")

    with open(os.path.join(model_dir, "config.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg = dacite.from_dict(config_lib.ModelConfig, cfg)
    if opts.size is None:
        opts.size = cfg.maze_size
    if opts.n_agents is None:
        opts.n_agents = cfg.model.max_agents

    session = tf.compat.v1.Session()
    model = agent.AgentModel(cfg.model)
    model_instance = model.build_layers()
    agent_fn = model.create_agent_fn(model_instance, session, True)

    model_instance.load_variables(session, opts.model_dir)

    results = []
    try:
        for _ in tqdm.tqdm(range(opts.n_games)):
            maze = maze_lib.generate_random_maze(opts.size, opts.size, 0.1)
            game = game_lib.Game.generate_random(maze, agent_fn, opts.n_agents)
            agent_fn.init_before_game(opts.n_agents)
            replays = game.play(opts.max_game_len or maze.height * maze.width * 2)

            results.append(list(map(len, replays)))
    except KeyboardInterrupt:
        return

    np.savetxt(opts.out, np.array(results), fmt="%d")


if __name__ == "__main__":
    main()
