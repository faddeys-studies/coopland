import tensorflow as tf
from coopland.maze_lib import Direction


def gather_present(vectors, present_indices):
    vectors = tf.pad(vectors, [(1, 0), (0, 0)])
    return tf.gather(vectors, present_indices + 1), tf.greater_equal(present_indices, 0)


directions_list = Direction.list_clockwise()
directions_to_i = {d: i for i, d in enumerate(directions_list)}
