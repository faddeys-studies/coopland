import numpy as np
import tensorflow as tf
from coopland.maze_lib import Direction
from coopland.tf_utils import get_shape_static_or_dynamic


def gather_present(vectors, present_indices, prepend_own=False):
    if prepend_own:
        n_total, max_others = get_shape_static_or_dynamic(present_indices)
        present_indices = tf.concat(
            [tf.expand_dims(tf.range(n_total), 1), present_indices], axis=1
        )
    vectors = tf.pad(vectors, [(1, 0), (0, 0)])
    return tf.gather(vectors, present_indices + 1), tf.greater_equal(present_indices, 0)


def build_visible_agents_features(
    comm_directions: "[batch time]", comm_distances: "[batch time]"
):
    comm_dirs_onehot = tf.one_hot(comm_directions, 4, dtype=tf.float32)
    comm_dists = tf.expand_dims(comm_distances, 2)
    return tf.concat([comm_dirs_onehot, comm_dists], axis=2)


def add_visible_agents_to_each_timestep(
    vectors: "[batch time vector]",
    comm_directions: "[batch time]",
    comm_distances: "[batch time]",
    prepend_own=False
):
    if prepend_own:
        comm_directions = tf.pad(comm_directions, [(0, 0), (1, 0)])
        comm_distances = tf.pad(comm_distances, [(0, 0), (1, 0)])
    return tf.concat(
        [build_visible_agents_features(comm_directions, comm_distances), vectors],
        axis=2,
    )


directions_list = Direction.list_clockwise()
directions_to_i = {d: i for i, d in enumerate(directions_list)}


class CommData:
    def __init__(self, n_agents, initial_buffer_size=64):
        self._times = np.zeros([n_agents], int)
        self._n_agents = n_agents
        self._max_others = 0
        self._buffer = self._make_buffer(initial_buffer_size)

    def _make_buffer(self, size=None):
        if size is None:
            size = self._buffer.shape[1]
        buf = np.zeros([self._n_agents, size, 3, self._n_agents], int)
        buf[...] = -1
        return buf

    def add_observation(self, observation):
        if observation is None:
            return
        vis_others = observation[3]
        agent_id = observation[0]
        t = self._times[agent_id]
        self._times[agent_id] += 1
        if not vis_others:
            return
        ids, dirs, dists = zip(*vis_others)
        dirs = [directions_to_i[d] for d in dirs]
        if t >= self._buffer.shape[1]:
            self._buffer = np.concatenate([self._buffer, self._make_buffer()], axis=1)
        n = len(ids)
        self._buffer[agent_id, t, :, :n] = ids, dirs, dists
        self._max_others = max(self._max_others, n)

    def build_batch(self):
        max_t = np.max(self._times)
        if max_t == 0:
            max_t = 1
        data = self._buffer[:, :max_t, :, : self._max_others]
        indices = data[:, :, 0, :]
        directions = data[:, :, 1, :]
        distances = data[:, :, 2, :].astype("float")
        return indices, directions, distances
