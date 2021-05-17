import os
import time
import tensorflow as tf

from tqdm import tqdm
from utils import utils
from model.model import MuGModel, create_optimizer, train_step, get_batch, tmp_build_model, compute_loss

songs = utils.get_songs_from_abc_dataset('data/irish_music.abc')

songs_text, vocabulary = utils.get_vocabulary(songs)
char2idx, idx2char = utils.get_lookup_tables(vocabulary)
vectorized_songs = utils.vectorize_string(songs_text, char2idx)

# Optimization parameters:
num_training_iterations = 4000
batch_size = 16
seq_length = 100
learning_rate = 5e-3

# Model parameters: 
vocab_size = len(vocabulary)
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './results'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = MuGModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 rnn_units=rnn_units,
                 batch_size=batch_size)

model.build((32,100))

# model = tmp_build_model(vocab_size=vocab_size,
#                         embedding_dim=embedding_dim,
#                         rnn_units=rnn_units,
#                         batch_size=batch_size)

optimizer = create_optimizer(learning_rate=learning_rate)

@tf.function
def tmp_train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

history = []
if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for iter in tqdm(range(200)):
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = tmp_train_step(x_batch, y_batch)

    history.append(loss.numpy().mean())

    if iter % 2:
        print(history[len(history) - 1])

    if iter % 100:
        model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)

#utils.play_audio_file('results/tmp.wav')
