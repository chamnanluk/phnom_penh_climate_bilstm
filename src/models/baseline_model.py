import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model


def build_baseline_bilstm(input_shape, learning_rate: float = 0.001):
    """Baseline multi-task BiLSTM without attention and without weighted loss."""
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    temp_output = Dense(1, activation="linear", name="temp_output")(x)
    rain_status_output = Dense(1, activation="sigmoid", name="rain_status_output")(x)
    rain_amount_output = Dense(1, activation="linear", name="rain_amount_output")(x)

    model = Model(inputs=inputs, outputs=[temp_output, rain_status_output, rain_amount_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "temp_output": "mse",
            "rain_status_output": "binary_crossentropy",
            "rain_amount_output": tf.keras.losses.Huber(),
        },
        loss_weights={
            "temp_output": 1.0,
            "rain_status_output": 1.0,
            "rain_amount_output": 2.0,
        },
        metrics={
            "temp_output": ["mae"],
            "rain_status_output": ["accuracy"],
            "rain_amount_output": ["mae"],
        },
    )
    return model
