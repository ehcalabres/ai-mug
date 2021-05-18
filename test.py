import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

import time
import tensorflow as tf

from tqdm import tqdm
from utils import utils
from model.model import MuGModel, generate_text

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
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

inference_input = 'X:1\n'

model = MuGModel(vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 rnn_units=rnn_units,
                 batch_size=batch_size)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

print("Model summary after restoring from checkpoint")

print(model.summary())

print("Generating new songs for the input: \'{}\'".format(inference_input))

inference_input_vectorized = utils.vectorize_string(inference_input, char2idx)

text_generated = generate_text(model, inference_input_vectorized)

final_text_generated = utils.reverse_vectorization(text_generated, idx2char)

with open('results/generated_songs.abc', 'w') as f:
    f.write(final_text_generated)