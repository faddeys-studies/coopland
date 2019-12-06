import tensorflow as tf
from coopland.maze_lib import Direction
from coopland.tf_utils import get_shape_static_or_dynamic


def gather_present(vectors, present_indices, prepend_own=False):
    if prepend_own:
        n_total, max_others = get_shape_static_or_dynamic(present_indices)
        present_indices = tf.concat([
            tf.expand_dims(tf.range(1, n_total+1), 1),
            present_indices,
        ], axis=1)
    vectors = tf.pad(vectors, [(1, 0), (0, 0)])
    return tf.gather(vectors, present_indices + 1), tf.greater_equal(present_indices, 0)


directions_list = Direction.list_clockwise()
directions_to_i = {d: i for i, d in enumerate(directions_list)}
