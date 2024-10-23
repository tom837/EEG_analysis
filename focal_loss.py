import tensorflow as tf



def focal_loss(gamma=2., alpha=.25):
    # Focal loss function used in model but not used in the end, was mostly for testing
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

        return tf.reduce_mean(loss, axis=1)

    return focal_loss_fixed