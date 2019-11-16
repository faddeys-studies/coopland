import logging
from coopland.game_lib import Direction
from coopland.models.a3c.training import TrainingContext, run_training


REWARD_NEW_EXPLORED = 0
REWARD_WIN = 1.0
REWARD_LOSE = -0.2
REWARD_STAY = 0
REWARD_REPEATED_VISIT = -0.1
REPEATED_VISIT_MIN = 2


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

        summaries_dir=".data/logs/try31",
        model_dir=".data/models/try31",
        do_visualize=True,
        per_game_callback=per_game_callback,

        system_supports_omp=True,
        omp_thread_limit=8,
        multithreaded_training=True,
        session_config=None,
    )
    run_training(ctx)


def reward_function(maze, replay, exit_pos):
    visited_points = {replay[0][1]: 1}
    seen_points = get_visible_positions(replay[0][0].observation[1], replay[0][1])
    total_points = maze.height * maze.width
    rewards = []
    for move, old_pos, new_pos in replay:
        r = 0.0
        if old_pos == new_pos:
            r += REWARD_STAY
        if new_pos in visited_points:
            # r -= 0.02 * visited_points[new_pos]
            visited_points[new_pos] += 1
        else:
            visited_points[new_pos] = 1
        if visited_points[new_pos] > REPEATED_VISIT_MIN:
            r += (visited_points[new_pos] - REPEATED_VISIT_MIN) * REWARD_REPEATED_VISIT
        visible_points = get_visible_positions(move.observation[1], old_pos)
        new_visible = visible_points - seen_points
        r += REWARD_NEW_EXPLORED * len(new_visible)
        seen_points.update(new_visible)
        if new_pos == exit_pos:
            r += REWARD_WIN
        rewards.append(r)
    if replay[-1][2] != exit_pos:
        rewards[-1] += REWARD_LOSE
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
