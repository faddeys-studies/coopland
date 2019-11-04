import tensorflow as tf
import numpy as np
import threading
import time
import os
import tqdm
import multiprocessing
import dataclasses
import logging
from queue import Queue, Full, Empty
from coopland.models.a3c.agent import AgentModel, AgentInstance
from coopland.game_lib import Game, Maze
from coopland.maze_lib import generate_random_maze
from coopland.visualizer_lib import VisualizerServer
from coopland import utils


@dataclasses.dataclass
class TrainingContext:
    # hyper parameters:
    discount_rate: float
    entropy_strength: float
    sync_each_n_games: int
    learning_rate: float
    actor_loss_weight: float
    critic_loss_weight: float
    reward_function: "(maze, replay, exit_pos) -> [N], where N - number of game steps"

    # infrastructure (where to save, visualization, debugging, etc):
    summaries_dir: str
    model_dir: str
    do_visualize: bool
    per_game_callback: "(...lots of data...) -> None"

    # performance tuning:
    system_supports_omp: bool
    omp_thread_limit: int
    multithreaded_training: bool
    session_config: tf.ConfigProto
    n_workers: int = None  # if None, then set automatically


class A3CWorker:
    def __init__(
        self,
        worker_id: int,
        session,
        model: AgentModel,
        global_instance: AgentInstance,
        optimizer: tf.compat.v1.train.Optimizer,
        greed: bool,
        train_context: TrainingContext,
    ):
        self.id = worker_id
        self.ctx = train_context
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
        _, actor_probs, critic_value, _, _ = self.instance.call(self.inputs_ph)

        weighted_actions = tf.reduce_sum(self.actions_ph * actor_probs, axis=-1)
        actor_loss_vector = -tf.math.log(weighted_actions + 1e-10) * self.advantage_ph
        actor_loss = tf.reduce_sum(actor_loss_vector)
        entropy = -tf.reduce_sum(actor_probs * tf.math.log(actor_probs + 1e-10))
        actor_full_loss = actor_loss
        if self.ctx.entropy_strength and self.ctx.entropy_strength > 0.0:
            actor_full_loss -= self.ctx.entropy_strength * entropy

        actor_gradients = tf.gradients(
            self.ctx.actor_loss_weight * actor_full_loss,
            self.instance.actor.trainable_variables,
        )
        actor_gradients = [tf.clip_by_norm(g, +5) for g in actor_gradients]
        actor_push_op = optimizer.apply_gradients(
            list(zip(actor_gradients, global_instance.actor.trainable_variables))
        )

        critic_loss = tf.reduce_mean(tf.square(self.reward_ph - critic_value))
        critic_gradients = tf.gradients(
            self.ctx.critic_loss_weight * critic_loss,
            self.instance.critic.trainable_variables,
        )
        critic_gradients = [tf.clip_by_norm(g, +5) for g in critic_gradients]
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
        self.agent_fn.init_before_game()
        replays = game.play(maze.height * maze.width * 3 // 2)
        replay = replays[0]

        immediate_reward = self.ctx.reward_function(maze, replay, game.exit_position)
        reward = discount(immediate_reward, self.ctx.discount_rate)
        critic_values = np.array([move.critic_value for move, _, _ in replay])
        advantage = reward - critic_values
        action_ids = np.array([move.direction_idx for move, _, _ in replay])
        actions_onehot = np.zeros([len(action_ids), 4])
        actions_onehot[np.arange(len(action_ids)), action_ids] = 1.0
        input_vectors = np.array([move.input_vector for move, _, _ in replay])

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

        if summary_writer is not None:
            results, summaries = results
            summary_writer.add_summary(summaries, game_index)

        if self.ctx.per_game_callback:
            self.ctx.per_game_callback(
                replays, immediate_reward, reward, critic_values, advantage
            )

        return game, replays

    def work_loop(self, mazes_queue, summary_writer, callback=None):
        last_epoch = -1
        while True:
            game_index, maze = mazes_queue.get()
            if maze is None:
                break
            log.info(f"W={self.id}: running #{game_index}")

            epoch = game_index // self.ctx.sync_each_n_games
            if epoch != last_epoch:
                self.session.run(self.fetch_op)
                last_epoch = epoch
            game, replays = self.work_on_one_game(maze, game_index, summary_writer)
            if callback:
                callback(game_index, game, replays)
            del maze, game, replays


def discount(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    next_r = discounted_rewards[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        r = rewards[i]
        next_r = gamma * next_r + r
        discounted_rewards[i] = next_r
    return discounted_rewards


def maze_generator_loop(mazes_queue, should_stop, initial_step, n_workers):
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
    for _ in range(n_workers):
        mazes_queue.put((None, None))


def run_training(train_context: TrainingContext):
    ctx = train_context

    total_internal_threads = (
        multiprocessing.cpu_count() if ctx.multithreaded_training else 1
    )
    if ctx.system_supports_omp:
        os.environ["OMP_THREAD_LIMIT"] = str(ctx.omp_thread_limit)
        n_workers = max(1, total_internal_threads // ctx.omp_thread_limit)
    else:
        n_workers = total_internal_threads
    if ctx.n_workers is None:
        ctx.n_workers = n_workers

    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.Session(config=ctx.session_config)
    model = AgentModel()
    global_inst = model.build_layers()
    optimizer = tf.compat.v1.train.RMSPropOptimizer(ctx.learning_rate)

    workers = [
        A3CWorker(
            i + 1,
            session,
            model,
            global_inst,
            optimizer,
            greed=(i == 0),
            train_context=ctx,
        )
        for i in tqdm.tqdm(range(ctx.n_workers), desc="Create workers")
    ]
    global_init_op = tf.compat.v1.variables_initializer(
        global_inst.get_variables() + optimizer.variables()
    )
    graph.finalize()
    log.info("graph finalized")

    session.run(global_init_op)

    if os.path.exists(ctx.model_dir):
        training_step = global_inst.load_variables(session, ctx.model_dir)
        log.info("restored model at step %s", training_step)
    else:
        training_step = 0
        os.makedirs(ctx.model_dir)

    summary_writer = tf.compat.v1.summary.FileWriter(ctx.summaries_dir)

    mazes_queue = Queue(maxsize=ctx.n_workers * 2)
    stop_event = threading.Event()
    vis_server = VisualizerServer(9876) if ctx.do_visualize else None
    all_threads = []

    def launch_thread(name, func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.setName(name)
        thread.start()
        all_threads.append(thread)
        return thread

    launch_thread(
        "mazegen",
        lambda: maze_generator_loop(
            mazes_queue, stop_event.isSet, training_step, ctx.n_workers
        ),
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
            time.sleep(600)
            with utils.interrupt_atomic():
                log.info("Saving model at step %s", training_step)
                global_inst.save_variables(session, ctx.model_dir, training_step)
    except KeyboardInterrupt:
        with utils.interrupt_atomic():
            log.info("Saving model at step %s", training_step)
            global_inst.save_variables(session, ctx.model_dir, training_step)
        log.info("\nInterrupted, shutting down gracefully")
        stop_event.set()
        for t in all_threads:
            log.info("joining %s", t.name)
            t.join()

    summary_writer.close()


log = logging.getLogger(__name__)
