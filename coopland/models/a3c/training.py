import tensorflow as tf
import numpy as np
import threading
import os
import tqdm
import multiprocessing
import logging
from queue import Queue, Full, Empty
from coopland.models.a3c.agent import AgentModel, AgentInstance
from coopland.models.a3c import data_utils, config_lib
from coopland.game_lib import Game, Maze
from coopland.maze_lib import generate_random_maze
from coopland.visualizer_lib import VisualizerServer
from coopland import utils


class A3CWorker:
    def __init__(
        self,
        worker_id: int,
        session,
        model: AgentModel,
        global_instance: AgentInstance,
        optimizer: tf.compat.v1.train.Optimizer,
        greed: bool,
        train_context: config_lib.TrainingContext,
    ):
        assert not train_context.training.use_data_augmentation
        self.id = worker_id
        self.ctx = train_context
        train_params = self.ctx.training
        self.model = model
        self.instance: AgentInstance = model.build_layers(name=f"Worker{worker_id}")
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
        batch_dim = self.ctx.problem.n_agents
        self.inputs_ph = tf.compat.v1.placeholder(
            tf.float32, [batch_dim, None, model.input_data_size]
        )
        self.actions_ph = tf.compat.v1.placeholder(tf.int32, [batch_dim, None])
        self.reward_ph = tf.compat.v1.placeholder(tf.float32, [batch_dim, None])
        self.advantage_ph = tf.compat.v1.placeholder(tf.float32, [batch_dim, None])
        self.episode_len_ph = tf.compat.v1.placeholder(tf.int32, [batch_dim])
        self.visible_others_ph = tf.compat.v1.placeholder(
            tf.int32, [batch_dim, None, None]
        )
        [
            actor_logits,
            actor_probs,
            critic_value,
            states_after,
            states_before_phs,
        ] = self.instance.call(
            self.inputs_ph,
            self.episode_len_ph,
            others_indices=self.visible_others_ph
            if self.model.hparams.use_communication
            else None,
        )
        del states_after
        del states_before_phs

        actor_loss_vector = (
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.actions_ph, logits=actor_logits
            )
            * self.advantage_ph
        )
        actor_loss = tf.reduce_sum(actor_loss_vector)
        entropy = -tf.reduce_sum(actor_probs * tf.math.log(actor_probs + 1e-10))
        actor_full_loss = actor_loss
        if train_params.entropy_strength and train_params.entropy_strength > 0.0:
            actor_full_loss -= train_params.entropy_strength * entropy

        critic_loss_vector = tf.square(self.reward_ph - critic_value)
        critic_loss = tf.reduce_mean(critic_loss_vector)
        reg_loss = 0.0
        if (
            train_params.regularization_strength
            and train_params.regularization_strength > 0.0
        ):
            reg_loss = train_params.regularization_strength * tf.add_n(
                [tf.reduce_sum(tf.square(v)) for v in self.instance.get_variables()]
            )

        total_loss = (
            train_params.actor_loss_weight * actor_full_loss
            + train_params.critic_loss_weight * critic_loss
            + reg_loss
        )

        gradients = tf.gradients(total_loss, self.instance.get_variables())
        gradients = [tf.clip_by_norm(g, +5) for g in gradients]
        self.push_op = optimizer.apply_gradients(
            list(zip(gradients, global_instance.get_variables()))
        )

        [action_logit_gradients] = tf.gradients(
            train_params.actor_loss_weight * actor_full_loss, [actor_logits]
        )

        _scalar = tf.compat.v1.summary.scalar
        _hist = tf.compat.v1.summary.histogram
        self.summary_op = tf.compat.v1.summary.merge(
            [
                _scalar("Performance/Reward", tf.reduce_mean(self.reward_ph)),
                _scalar("Performance/Advantage", tf.reduce_mean(self.advantage_ph)),
                _scalar("Performance/NSteps", tf.shape(self.actions_ph)[1]),
                _scalar(
                    "Performance/NSteps_mean",
                    tf.reduce_mean(tf.cast(self.episode_len_ph, tf.float32)),
                ),
                _scalar(
                    "Performance/Communications",
                    tf.reduce_sum(tf.cast(self.visible_others_ph >= 0, tf.float32)) / 2,
                ),
                _scalar("Train/Entropy", entropy),
                _scalar("Train/TotalLoss", total_loss),
                _scalar("Train/ActorLoss", actor_loss),
                _hist("Train/ActorLoss_hist", actor_loss_vector),
                _scalar("Train/CriticLoss", critic_loss),
                _hist("Train/CriticLoss_hist", critic_loss_vector),
                _scalar("Train/ActorFullLoss", actor_full_loss),
                *[
                    _hist(
                        f"Performance/Actions/{d}",
                        tf.cast(tf.equal(self.actions_ph, i), tf.float32),
                    )
                    for d, i in self.model.directions_to_i.items()
                ],
                *[
                    _scalar(f"ActorLogits/{d}", tf.reduce_mean(actor_logits[:, :, i]))
                    for d, i in self.model.directions_to_i.items()
                ],
                *[
                    _hist(f"ActorLogits/{d}_hist", actor_logits[:, :, i])
                    for d, i in self.model.directions_to_i.items()
                ],
                *[
                    _scalar(
                        f"ActorGradients/{d}",
                        tf.reduce_mean(action_logit_gradients[:, :, i]),
                    )
                    for d, i in self.model.directions_to_i.items()
                ],
                *[
                    _hist(f"ActorGradients/{d}_hist", action_logit_gradients[:, :, i])
                    for d, i in self.model.directions_to_i.items()
                ],
            ]
        )
        self.push_op_and_summaries = self.push_op, self.summary_op

    def work_on_one_game(self, maze: Maze, game_index, summary_writer):
        game = Game.generate_random(maze, self.agent_fn, self.ctx.problem.n_agents)
        self.agent_fn.init_before_game()
        replays = game.play(maze.height * maze.width * 3 // 2)
        immediate_rewards = self.ctx.problem.reward_function(
            maze, replays, game.exit_position
        )
        rewards = []
        critic_values = []
        advantages = []
        inputs_batch = []
        actions_batch = []
        visible_others_batch = []

        def padseq(x, value=0.0, where="post"):
            return tf.keras.preprocessing.sequence.pad_sequences(
                x, dtype=x[0].dtype, padding=where, value=value
            )

        for replay, immediate_reward in zip(replays, immediate_rewards):
            reward = discount(immediate_reward, self.ctx.training.discount_rate)
            critic_value = np.array([move.critic_value for move, _, _ in replay])
            advantage = reward - critic_value
            input_vectors, action_ids = data_utils.get_training_batch(replay)
            visible_others = padseq(
                [
                    np.array([ag_id for ag_id, _, _ in move.observation[3]] or [-1])
                    for move, _, _ in replay
                ],
                value=-1, where="pre"
            )

            rewards.append(reward)
            critic_values.append(critic_value)
            advantages.append(advantage)
            inputs_batch.append(input_vectors[0])
            actions_batch.append(action_ids[0])
            visible_others_batch.append(visible_others)
        max_t, max_vo = np.max([vo.shape for vo in visible_others_batch], axis=0)
        visible_others_batch = np.array([
            np.pad(vo, [(0, max_t-vo.shape[0]), (max_vo - vo.shape[1], 0)],
                   mode='constant', constant_values=-1)
            for vo in visible_others_batch
        ])

        op = self.push_op if summary_writer is None else self.push_op_and_summaries

        results = self.session.run(
            op,
            {
                self.reward_ph: padseq(rewards),
                self.advantage_ph: padseq(advantages),
                self.actions_ph: padseq(actions_batch),
                self.inputs_ph: padseq(inputs_batch),
                self.visible_others_ph: visible_others_batch,
                self.episode_len_ph: [len(replay) for replay in replays],
            },
        )

        if summary_writer is not None:
            results, summaries = results
            summary_writer.add_summary(summaries, game_index)

        if self.ctx.infrastructure.per_game_callback:
            self.ctx.infrastructure.per_game_callback(
                replays, immediate_rewards, rewards, critic_values, advantages
            )

        return game, replays

    def work_loop(self, mazes_queue, summary_writer, callback=None):
        last_epoch = -1
        while True:
            game_index, maze = mazes_queue.get()
            if maze is None:
                break
            log.info(f"W={self.id}: running #{game_index}")

            epoch = game_index // self.ctx.training.sync_each_n_games
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


def maze_generator_loop(
    mazes_queue, maze_size, should_stop, initial_step, end_step, n_workers
):
    i = initial_step
    while not should_stop():
        i += 1
        if end_step is not None and i > end_step:
            break
        maze = generate_random_maze(*maze_size, 0.01)
        while not should_stop():
            try:
                mazes_queue.put((i, maze), timeout=1)
            except Full:
                pass
            else:
                break
    if should_stop():
        try:
            while True:
                mazes_queue.get_nowait()
        except Empty:
            pass
    for _ in range(n_workers):
        mazes_queue.put((None, None))


def run_training(train_context: config_lib.TrainingContext):
    ctx = train_context
    perf = train_context.performance
    infr = ctx.infrastructure

    total_internal_threads = (
        multiprocessing.cpu_count() if perf.multithreaded_training else 1
    )
    if perf.system_supports_omp:
        os.environ["OMP_THREAD_LIMIT"] = str(perf.omp_thread_limit)
        n_workers = max(1, total_internal_threads // perf.omp_thread_limit)
    else:
        n_workers = total_internal_threads
    if perf.n_workers is None:
        perf.n_workers = n_workers

    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.Session(config=perf.session_config)
    model = AgentModel(ctx.model)
    global_inst = model.build_layers()
    optimizer = tf.compat.v1.train.RMSPropOptimizer(ctx.training.learning_rate)

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
        for i in tqdm.tqdm(range(perf.n_workers), desc="Create workers")
    ]
    global_init_op = tf.compat.v1.variables_initializer(
        global_inst.get_variables() + optimizer.variables()
    )
    graph.finalize()
    log.info("graph finalized")

    session.run(global_init_op)

    if tf.train.latest_checkpoint(infr.model_dir) is not None:
        training_step = global_inst.load_variables(session, infr.model_dir)
        log.info("restored model at step %s", training_step)
    else:
        training_step = 0
        os.makedirs(infr.model_dir, exist_ok=True)

    summary_writer = tf.compat.v1.summary.FileWriter(infr.summaries_dir)

    mazes_queue = Queue(maxsize=perf.n_workers * 2)
    stop_event = threading.Event()
    lock = threading.Lock()
    vis_server = VisualizerServer(9876) if infr.do_visualize else None
    all_threads = []

    def launch_thread(name, func, *args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.setName(name)
        thread.start()
        all_threads.append(thread)
        return thread

    mazegen_thread = launch_thread(
        "mazegen",
        lambda: maze_generator_loop(
            mazes_queue,
            ctx.problem.maze_size,
            stop_event.isSet,
            training_step,
            infr.train_until_n_games,
            perf.n_workers,
        ),
    )

    def worker_callback(game_id, game, replays):
        nonlocal training_step
        if vis_server:
            vis_server.add_replay(game_id, game, replays)
        with lock:
            training_step = max(game_id, training_step)

    for i, worker in enumerate(workers):
        launch_thread(
            f"worker_{i}",
            worker.work_loop,
            mazes_queue,
            summary_writer if i == 0 else None,
            callback=worker_callback,
        )

    try:
        while mazegen_thread.is_alive():
            mazegen_thread.join(600)
            if not mazegen_thread.is_alive():
                break
            with utils.interrupt_atomic():
                log.info("Saving model at step %s", training_step)
                global_inst.save_variables(session, infr.model_dir, training_step)
    except KeyboardInterrupt:
        log.info("\nInterrupted")
    log.info("Shutting down gracefully")
    stop_event.set()
    for t in all_threads:
        log.info("joining %s", t.name)
        t.join()
    with utils.interrupt_atomic():
        log.info("Saving model at step %s", training_step)
        global_inst.save_variables(session, infr.model_dir, training_step)

    summary_writer.close()


log = logging.getLogger(__name__)
