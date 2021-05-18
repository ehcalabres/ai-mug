import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import time
import tensorflow as tf

from tqdm import tqdm
from utils import utils
from model.model import MuGModel, create_optimizer, generate_text, train_step, get_batch, compute_loss

songs = utils.get_songs_from_abc_dataset('data/irish_music.abc')

songs_text, vocabulary = utils.get_vocabulary(songs)
char2idx, idx2char = utils.get_lookup_tables(vocabulary)
vectorized_songs = utils.vectorize_string(songs_text, char2idx)

num_training_iterations = 4000
batch_size = 16
seq_length = 100
learning_rate = 5e-3
 
vocab_size = len(vocabulary)
embedding_dim = 256
rnn_units = 1024
 
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = MuGModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 rnn_units=rnn_units,
                 batch_size=batch_size)

model.build((batch_size, seq_length))

print(model.summary())

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
pbar = tqdm(range(num_training_iterations))
for iter in pbar:
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(model, optimizer, x_batch, y_batch)

    history.append(loss.numpy().mean())

    if iter % 100:
        model.save_weights(checkpoint_prefix)

    pbar.set_postfix({'loss': history[len(history) - 1]})

model.save_weights(checkpoint_prefix)

inference_input = 'X:1\nK: D major\n'

model = MuGModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 rnn_units=rnn_units,
                 batch_size=batch_size)

model.build(tf.TensorShape([1, None]))
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

print("Model summary after restoring from checkpoint")

print(model.summary())

print("Generating new songs for the input: \'{}\'".format(inference_input))

inference_input_vectorized = utils.vectorize_string(inference_input, char2idx)

text_generated = generate_text(model, inference_input_vectorized)

final_text_generated = utils.reverse_vectorization(text_generated, idx2char)

with open('results/generated_songs.abc', 'w') as f:
    f.write(final_text_generated)

