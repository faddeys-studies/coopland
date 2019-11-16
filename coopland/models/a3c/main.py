import logging
from coopland.game_lib import Direction
from coopland.models.a3c.training import TrainingContext, run_training


def main():
    logging.basicConfig(level=logging.INFO)
    ctx = TrainingContext(
        discount_rate=0.9,
        entropy_strength=0.0003,
        sync_each_n_games=1,
        learning_rate=0.001,
        actor_loss_weight=1.0,
        critic_loss_weight=0.25,
        use_data_augmentation=False,
        reward_function=reward_function,
        regularization_strength=-1,

        summaries_dir=".data/logs/rew2-2",
        model_dir=".data/models/rew2-2",
        do_visualize=True,
        per_game_callback=per_game_callback,

        system_supports_omp=True,
        omp_thread_limit=8,
        multithreaded_training=True,
        session_config=None,
    )
    run_training(ctx)


def reward_function(maze, replay, exit_pos):
    distances = {}

    _q = [(exit_pos, 0)]
    while _q:
        pos, dist = _q.pop(0)
        if pos not in distances or dist < distances[pos]:
            distances[pos] = dist
            for d in _directions:
                if maze.has_path(*pos, direction=d):
                    next_pos = d.apply(*pos)
                    if next_pos not in distances or dist+1 < distances[next_pos]:
                        _q.append((next_pos, dist+1))

    rewards = []
    for move, old_pos, new_pos in replay:
        d_old = distances[old_pos]
        d_new = distances[new_pos]
        r = 0.5 * (d_old - d_new)
        rewards.append(r)
    if replay[-1][2] == exit_pos:
        rewards[-1] += 1.0
    return rewards


def per_game_callback(replays, immediate_reward, reward, critic_values, advantage):
    for (move, _, _), r_i, r_d, v, a in zip(
        replays[0], immediate_reward, reward, critic_values, advantage
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


if __name__ == "__main__":
    main()
