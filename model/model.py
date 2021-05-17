import numpy as np
import tensorflow as tf

class MuGModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size) -> None:
        super(MuGModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[batch_size, None])
        self.lstm_layer_1 = lstm(rnn_units)
        self.lstm_layer_2 = lstm(rnn_units // 4)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.lstm_layer_1(x)
        x = self.lstm_layer_2(x)
        x = self.output(x)
        return x

def tmp_build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        lstm(rnn_units),
        lstm(rnn_units // 4),
        tf.keras.layers.Dense(vocab_size)
    ])

def lstm(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True
    )

def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    return loss

def create_optimizer(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def get_batch(vectorized_songs, seq_length, batch_size):
  idx = np.random.choice(vectorized_songs.shape[0] - 1 - seq_length, batch_size)

  input_batch = np.array([vectorized_songs[i : i + seq_length] for i in idx])
  output_batch = np.array([vectorized_songs[i + 1 : i + seq_length + 1] for i in idx])

  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])

  return x_batch, y_batch

def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model, optimizer, loss