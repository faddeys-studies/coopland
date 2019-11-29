import argparse
import os
import yaml
import dacite
import tensorflow as tf
from coopland.a3c import agent
from coopland.visualizer_lib import Visualizer
from coopland import game_lib, maze_lib, config_lib


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    cli.add_argument("--size", type=int)
    cli.add_argument("--n-agents", type=int)
    opts = cli.parse_args()

    model_dir = opts.model_dir

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

    visualizer = Visualizer(
        cell_size_px=100,
        sec_per_turn=0.5,
        move_animation_sec=0.4,
        autoplay=True,
        autoend=False,
    )

    try:
        while True:
            maze = maze_lib.generate_random_maze(opts.size, opts.size, 0.1)
            game = game_lib.Game.generate_random(maze, agent_fn, opts.n_agents)
            agent_fn.init_before_game()
            replays = game.play(maze.height * maze.width * 2)

            visualizer.run(game, replays)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
