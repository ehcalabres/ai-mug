import time
import librosa
import threading
import regex as re
import numpy as np

from tqdm import tqdm
from typing import List, overload
from playsound import playsound

def play_audio_file(filepath, step=1):
    duration = round(librosa.get_duration(filename=filepath), 2)

    ret = [None]
    def my_runner(playsound, ret):
        ret[0] = playsound(filepath)

    thread = threading.Thread(target=my_runner, args=(playsound, ret))
    bar = tqdm(total=duration)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=step)
        bar.update(step)
    
    bar.close()

def get_songs_from_abc_dataset(filepath):
    with open(filepath, "r") as f:
        dataset_text = f.read()
    
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, dataset_text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text dataset.".format(len(songs)))
    return songs

def get_vocabulary(texts):
    texts_joined = "\n\n".join(texts)
    vocabulary = sorted(set(texts_joined))
    return texts_joined, vocabulary

def get_lookup_tables(vocabulary):
    char2idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx2char = np.array(vocabulary)
    return char2idx, idx2char

def vectorize_string(string, char2idx):
    char_vector = [char2idx[char] for char in string]
    return np.array(char_vector)

