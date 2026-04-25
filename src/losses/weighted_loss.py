import tensorflow as tf


def make_weighted_huber_loss(scaled_thresholds=None, threshold_weights=None, delta=1.0):
    """
    Weighted Huber loss for rainfall amount.

    y_true is scaled log-rainfall. Thresholds must also be in scaled log-rainfall units.
    Higher thresholds get larger weights so the model pays more attention to heavy rain.
    """
    if scaled_thresholds is None:
        scaled_thresholds = []
    if threshold_weights is None:
        threshold_weights = []

    scaled_thresholds = [float(x) for x in scaled_thresholds]
    threshold_weights = [float(x) for x in threshold_weights]

    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + delta * linear

        weights = tf.ones_like(y_true)
        for threshold, weight in zip(scaled_thresholds, threshold_weights):
            weights = tf.where(y_true >= threshold, tf.ones_like(y_true) * weight, weights)

        return tf.reduce_mean(huber * weights)

    return loss
