import tensorflow as tf
import numpy as np
import threading
import time
import os
import psutil
import tqdm
import multiprocessing
import gc
from queue import Queue, Full, Empty
from coopland.models.a3c.agent import AgentModel, AgentInstance
from coopland.game_lib import Game, Maze, Direction
from coopland.maze_lib import generate_random_maze
from coopland.visualizer_lib import VisualizerServer
from coopland import utils


TOTAL_INTERNAL_THREADS = multiprocessing.cpu_count()
THREADS_PER_WORKER = 35
N_WORKERS = TOTAL_INTERNAL_THREADS // THREADS_PER_WORKER
DISCOUNT_RATE = 1.00
ENTROPY_STRENGTH = 0.1
SUMMARIES_DIR = ".data/logs/test4"
MODEL_DIR = ".data/models/test4"
VISUALIZE = True
os.environ["OMP_THREAD_LIMIT"] = str(THREADS_PER_WORKER)


_last_memory_use = 0.0


def memory(when):
    global _last_memory_use
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # memory_use = py.memory_info()[0] / 2.0 ** 20
    # if abs(1000000 * (memory_use - _last_memory_use)) > 1:
    #     print(f"memory: {when}: {memory_use:4f} d={memory_use - _last_memory_use:+4f}")
    #     _last_memory_use = memory_use


class A3CWorker:
    def __init__(
        self,
        worker_id: int,
        session,
        model: AgentModel,
        global_instance: AgentInstance,
        optimizer: tf.compat.v1.train.Optimizer,
        greed: bool,
    ):
        self.id = worker_id
        self.model = model
        self.instance = model.build_layers(name=f"Worker{worker_id}")
        self.global_instance = global_instance
        self.agent_fn = model.create_agent_fn(
            self.instance, session, 1.001 if greed else None
        )
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
        actor_loss = tf.reduce_sum(actor_loss_vector)
        entropy = -tf.reduce_sum(actor_probs * tf.math.log(actor_probs + 1e-10))
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
        game = Game.generate_random(maze, self.agent_fn, 1)
        memory("before init")
        self.session.run(self.fetch_op)
        memory("fetch op")
        self.agent_fn.init_before_game()
        replays = game.play(maze.height * maze.width * 3 // 2)
        memory("game played")
        replay = replays[0]

        immediate_reward = build_immediate_reward(maze, replay, game.exit_position)
        reward = discount(immediate_reward, DISCOUNT_RATE)
        critic_values = np.array([move.critic_value for move, _, _ in replay])
        advantage = reward - critic_values
        action_ids = np.array([move.direction_idx for move, _, _ in replay])
        actions_onehot = np.zeros([len(action_ids), 4])
        actions_onehot[np.arange(len(action_ids)), action_ids] = 1.0
        input_vectors = np.array([move.input_vector for move, _, _ in replay])
        memory("train inputs calculated")

        op = self.push_op if summary_writer is None else self.push_op_and_summaries

        results = self.session.run(
            op,
            {
                self.reward_ph: [reward],
                self.advantage_ph: [advantage],
                self.actions_ph: [actions_onehot],
                self.inputs_ph: [input_vectors],
            },
        )
        memory("train step done")

        if summary_writer is not None:
            results, summaries = results
            summary_writer.add_summary(summaries, game_index)
            memory("summary written")

        for (move, _, _), r_i, r_d, v, a in zip(
            replays[0], immediate_reward, reward, critic_values, advantage
        ):
            move.debug_text += (
                f"{move.direction.upper()[:1]} "
                f"({move.probabilities[move.direction_idx]:.2f})"
                f" V={v:.2f}\n"
                f"Ri={r_i:.2f} "
                f"Rd={r_d:.2f} "
                f"A={a:.2f}"
            )

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


def build_immediate_reward(maze, replay, exit_pos):
    visited_points = {replay[0][1]: 1}
    seen_points = get_visible_positions(replay[0][0].observation[1], replay[0][1])
    total_points = maze.height * maze.width
    rewards = []
    for move, _, new_pos in replay:
        r = 0.0
        if new_pos in visited_points:
            # r -= 0.02 * visited_points[new_pos]
            visited_points[new_pos] += 1
        else:
            visited_points[new_pos] = 1
        visible_points = get_visible_positions(move.observation[1], new_pos)
        new_visible = visible_points - seen_points
        move.debug_text += f"{new_pos} {visible_points} {move.observation[1]}\n"
        r += 0.5 * len(new_visible)
        seen_points.update(new_visible)
        if new_pos == exit_pos:
            r += 10 * (1 + len(seen_points) / total_points) / 2
        rewards.append(r)
    if replay[-1][2] != exit_pos:
        rewards[-1] -= 10
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


def maze_generator_loop(mazes_queue, should_stop, initial_step):
    i = initial_step
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
    try:
        while True:
            mazes_queue.get_nowait()
    except Empty:
        pass
    for _ in range(N_WORKERS):
        mazes_queue.put((None, None))


def main():
    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.Session(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=N_WORKERS,
            intra_op_parallelism_threads=1,
            allow_soft_placement=True,
        )
    )
    model = AgentModel()
    global_inst = model.build_layers()
    optimizer = tf.compat.v1.train.RMSPropOptimizer(0.01)

    workers = [
        A3CWorker(i + 1, session, model, global_inst, optimizer, greed=(i == 0))
        for i in tqdm.tqdm(range(N_WORKERS), desc="Create workers")
    ]
    global_init_op = tf.compat.v1.variables_initializer(
        global_inst.get_variables() + optimizer.variables()
    )
    graph.finalize()
    print("graph finalized")

    session.run(global_init_op)

    if os.path.exists(MODEL_DIR):
        training_step = global_inst.load_variables(session, MODEL_DIR)
        print("restored model at step", training_step)
    else:
        training_step = 0
        os.makedirs(MODEL_DIR)

    summary_writer = tf.compat.v1.summary.FileWriter(SUMMARIES_DIR)

    mazes_queue = Queue(maxsize=N_WORKERS * 2)
    stop_event = threading.Event()
    vis_server = VisualizerServer(9876) if VISUALIZE else None
    all_threads = []

    def launch_thread(name, func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.setName(name)
        thread.start()
        all_threads.append(thread)
        return thread

    launch_thread(
        "mazegen",
        lambda: maze_generator_loop(mazes_queue, stop_event.isSet, training_step),
    )

    def worker_callback(game_id, game, replays):
        nonlocal training_step
        if vis_server:
            vis_server.add_replay(game_id, game, replays)
        training_step = game_id

    for i, worker in enumerate(workers):
        launch_thread(
            f"worker_{i}",
            worker.work_loop,
            mazes_queue,
            summary_writer if i == 0 else None,
            callback=worker_callback,
        )

    try:
        while True:
            # print(f"replays_q={replays_queue.qsize()} "
            #       f"mazes_q={mazes_queue.qsize()} "
            #       f"summary_writer={summary_writer.event_writer._event_queue.qsize()} "
            #       f"n_threads={threading.active_count()}/{len(all_threads)} "
            #       f"")
            time.sleep(600)
            with utils.interrupt_atomic():
                print("Saving model at step", training_step)
                global_inst.save_variables(session, MODEL_DIR, training_step)
    except KeyboardInterrupt:
        with utils.interrupt_atomic():
            print("Saving model at step", training_step)
            global_inst.save_variables(session, MODEL_DIR, training_step)
        print("\nInterrupted, shutting down gracefully")
        stop_event.set()
        for t in all_threads:
            print("joining", t.name)
            t.join()

    summary_writer.close()


if __name__ == "__main__":
    main()
