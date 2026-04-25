import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.models import Model

from src.losses.weighted_loss import make_weighted_huber_loss


class AttentionLayer(Layer):
    """Simple temporal attention layer for BiLSTM sequence outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context


def build_attention_bilstm(
    input_shape,
    learning_rate: float = 0.001,
    scaled_thresholds=None,
    threshold_weights=None,
):
    """Next-level model: BiLSTM + temporal attention + weighted rainfall loss."""
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = AttentionLayer()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    temp_output = Dense(1, activation="linear", name="temp_output")(x)
    rain_status_output = Dense(1, activation="sigmoid", name="rain_status_output")(x)
    rain_amount_output = Dense(1, activation="linear", name="rain_amount_output")(x)

    model = Model(inputs=inputs, outputs=[temp_output, rain_status_output, rain_amount_output])

    rain_loss = make_weighted_huber_loss(
        scaled_thresholds=scaled_thresholds,
        threshold_weights=threshold_weights,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "temp_output": "mse",
            "rain_status_output": "binary_crossentropy",
            "rain_amount_output": rain_loss,
        },
        loss_weights={
            "temp_output": 0.8,
            "rain_status_output": 1.0,
            "rain_amount_output": 4.0,
        },
        metrics={
            "temp_output": ["mae"],
            "rain_status_output": ["accuracy"],
            "rain_amount_output": ["mae"],
        },
    )
    return model
