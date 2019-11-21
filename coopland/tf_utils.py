import tensorflow as tf


def get_shape_static_or_dynamic(tensor: tf.Tensor):
    shape = tensor.get_shape().as_list()
    if None in shape:
        shape_dynamic = tf.shape(tensor)
        shape = [x if x is not None else shape_dynamic[i]
                 for i, x in enumerate(shape)]
    return shape
