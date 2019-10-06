import tensorflow as tf
import numpy as np
import threading
import time
import os
import psutil
import gc
from queue import Queue, Full
from coopland.models.a3c.agent import AgentModel, AgentInstance
from coopland.game_lib import Game, Maze, Direction
from coopland.maze_lib import generate_random_maze
from coopland.visualizer_lib import Visualizer


DISCOUNT_RATE = 0.8
ENTROPY_STRENGTH = 0.1
SUMMARIES_DIR = ".data/logs/try7"
N_WORKERS = 2


_last_memory_use = 0.0


def memory(when):
    global _last_memory_use
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 20
    if abs(1000000 * (memory_use - _last_memory_use)) > 1:
        print(f"memory: {when}: {memory_use:4f} d={memory_use - _last_memory_use:+4f}")
        _last_memory_use = memory_use


class A3CWorker:
    def __init__(
        self,
        worker_id: int,
        session,
        model: AgentModel,
        global_instance: AgentInstance,
        optimizer: tf.compat.v1.train.Optimizer,
    ):
        self.id = worker_id
        self.model = model
        self.instance = model.build_layers(name=f"Worker{worker_id}")
        self.global_instance = global_instance
        self.agent_fn = model.create_agent_fn(self.instance, session)
        self.session = session

        local_variables = self.instance.get_variables()
        global_variables = global_instance.get_variables()

        self.fetch_op = [
            tf.compat.v1.assign(local_var, global_var)
            for local_var, global_var in zip(local_variables, global_variables)
        ]
        self.inputs_ph = tf.compat.v1.placeholder(
            tf.float32, [1, None, AgentModel.INPUT_DATA_SIZE]
        )
        self.actions_ph = tf.compat.v1.placeholder(tf.float32, [1, None, 4])
        self.reward_ph = tf.compat.v1.placeholder(tf.float32, [1, None])
        self.advantage_ph = tf.compat.v1.placeholder(tf.float32, [1, None])
        _, actor_probs, critic_value, _, _ = self.instance(self.inputs_ph)

        weighted_actions = tf.reduce_sum(self.actions_ph * actor_probs, axis=-1)
        actor_loss_vector = -tf.math.log(weighted_actions + 1e-10) * self.advantage_ph
        entropy = -tf.reduce_sum(actor_probs * tf.math.log(actor_probs + 1e-10))
        actor_loss = tf.reduce_sum(actor_loss_vector)
        actor_full_loss = actor_loss - ENTROPY_STRENGTH * entropy

        actor_gradients = tf.gradients(
            actor_full_loss, self.instance.actor.trainable_variables
        )
        actor_gradients = [tf.clip_by_value(g, -5, +5) for g in actor_gradients]
        actor_push_op = optimizer.apply_gradients(
            list(zip(actor_gradients, global_instance.actor.trainable_variables))
        )

        critic_loss = tf.reduce_mean(tf.square(self.reward_ph - critic_value))
        critic_gradients = tf.gradients(
            critic_loss, self.instance.critic.trainable_variables
        )
        critic_gradients = [tf.clip_by_value(g, -5, +5) for g in critic_gradients]
        critic_push_op = optimizer.apply_gradients(
            list(zip(critic_gradients, global_instance.critic.trainable_variables))
        )

        self.push_op = actor_push_op, critic_push_op

        _scalar = tf.compat.v1.summary.scalar
        _hist = tf.compat.v1.summary.histogram
        self.summary_op = tf.compat.v1.summary.merge(
            [
                _scalar("Performance/Reward", tf.reduce_mean(self.reward_ph)),
                _scalar("Performance/Advantage", tf.reduce_mean(self.advantage_ph)),
                _scalar("Performance/NSteps", tf.shape(self.actions_ph)[1]),
                # _hist(
                #     "Performance/Actions", tf.reduce_mean(self.actions_ph, axis=(0, 1))
                # ),
                _scalar("Train/Entropy", entropy),
                _scalar("Train/ActorLoss", actor_loss),
                _scalar("Train/CriticLoss", critic_loss),
                _scalar("Train/ActorFullLoss", actor_full_loss),
            ]
        )
        self.push_op_and_summaries = self.push_op, self.summary_op

    def work_on_one_game(self, maze: Maze, game_index, summary_writer):
        game = Game(maze, self.agent_fn, 1)
        memory("before init")
        self.session.run(self.fetch_op)
        memory("fetch op")
        self.agent_fn.init_before_game()
        replays = game.play(maze.height * maze.width * 3 // 2)
        memory("game played")
        replay = replays[0]

        reward = discount(
            build_immediate_reward(replay, game.exit_position), DISCOUNT_RATE
        )
        critic_values = np.array([move.critic_value for move, _, _ in replay])
        advantage = reward - critic_values
        action_ids = np.array([move.direction_idx for move, _, _ in replay])
        actions_onehot = np.zeros([len(action_ids), 4])
        actions_onehot[np.arange(len(action_ids)), action_ids] = 1.0
        input_vectors = np.array([move.input_vector for move, _, _ in replay])
        memory("train inputs calculated")

        op = self.push_op if summary_writer is None else self.push_op_and_summaries

        results = self.session.run(op, {
            self.reward_ph: [reward],
            self.advantage_ph: [advantage],
            self.actions_ph: [actions_onehot],
            self.inputs_ph: [input_vectors],
        })
        memory("train step done")

        if summary_writer is not None:
            results, summaries = results
            summary_writer.add_summary(summaries, game_index)
            memory("summary written")

        return game, replays

    def work_loop(self, mazes_queue, summary_writer, callback=None):
        graph = tf.get_default_graph()
        if graph is not None and not graph.finalized:
            graph.finalize()
        while True:
            game_index, maze = mazes_queue.get()
            if maze is None:
                break
            print(f"W={self.id}: running #{game_index}")
            game, replays = self.work_on_one_game(maze, game_index, summary_writer)
            if callback:
                callback(game_index, game, replays)
            del maze, game, replays
            memory("game done")
            gc.collect()
            memory("gc")


def build_immediate_reward(replay, exit_pos):
    visited_points = {replay[0][1]}
    seen_points = get_visible_positions(replay[0][0].observation[1], replay[0][1])
    rewards = []
    for move, _, new_pos in replay:
        r = 0.0
        if new_pos in visited_points:
            r -= 0.1
        else:
            visited_points.add(new_pos)
        visible_points = get_visible_positions(move.observation[1], new_pos)
        new_visible = visible_points - seen_points
        r += 0.1 * len(new_visible)
        seen_points.update(new_visible)
        if new_pos == exit_pos:
            r += 10
        rewards.append(r)
    return rewards


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


def maze_generator_loop(mazes_queue, should_stop):
    i = 0
    while not should_stop():
        maze = generate_random_maze(4, 4, 0.01)
        i += 1
        while not should_stop():
            try:
                mazes_queue.put((i, maze), timeout=1)
            except Full:
                pass
            else:
                break
    mazes_queue.put((None, None))


def main():
    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.Session(
        config=tf.ConfigProto(inter_op_parallelism_threads=N_WORKERS)
    )
    model = AgentModel()
    global_inst = model.build_layers()
    optimizer = tf.compat.v1.train.RMSPropOptimizer(0.01)

    workers = [
        A3CWorker(i + 1, session, model, global_inst, optimizer)
        for i in range(N_WORKERS)
    ]
    global_init_op = tf.compat.v1.variables_initializer(
        global_inst.get_variables() + optimizer.variables()
    )
    graph.finalize()
    print("graph finalized")

    session.run(global_init_op)

    summary_writer = tf.compat.v1.summary.FileWriter(SUMMARIES_DIR)

    mazes_queue = Queue(maxsize=N_WORKERS * 2)
    replays_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    # visualizer = Visualizer(
    #     cell_size_px=50, sec_per_turn=0.5, move_animation_sec=0.4, autoplay=True
    # )
    all_threads = []

    def launch_thread(name, func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.setName(name)
        thread.start()
        all_threads.append(thread)
        return thread

    launch_thread("mazegen", lambda: maze_generator_loop(mazes_queue, stop_event.isSet))

    def add_replay(*item):
        try:
            replays_queue.put_nowait(item)
        except Full:
            pass

    for i, worker in enumerate(workers):
        launch_thread(
            f"worker_{i}",
            worker.work_loop,
            mazes_queue,
            summary_writer if i == 0 else None,
            callback=None,  # add_replay
        )

    try:
        # while True:
        #     game_index, game, replays = replays_queue.get()
        #     visualizer.title = f"game #{game_index}"
        #     visualizer.run(game, replays)
        while True:
            # print(f"replays_q={replays_queue.qsize()} "
            #       f"mazes_q={mazes_queue.qsize()} "
            #       f"summary_writer={summary_writer.event_writer._event_queue.qsize()} "
            #       f"n_threads={threading.active_count()}/{len(all_threads)} "
            #       f"")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down gracefully")
        stop_event.set()
        for t in all_threads:
            print("joining", t.name)
            t.join()

    summary_writer.close()


if __name__ == "__main__":
    main()
