import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def load_preprocessed_data(data_path, shuffle_data=True):
  
    with open(f"{data_path}/idx_to_word.pickle", "rb") as idx_to_word_in:
        idx_to_word = pickle.load(idx_to_word_in)

    with open(f"{data_path}/word_to_idx.pickle", "rb") as word_to_index_in:
        word_to_idx = pickle.load(word_to_index_in)

    freqs = np.load(f"{data_path}/freqs.npy").tolist()

    df = pd.read_csv(f"{data_path}/skipgrams.txt", sep="\t", header=None)

    pivot_ids = df[0].values
    target_ids = df[1].values
    doc_ids = df[2].values

    if shuffle_data:
        pivot_ids, target_ids, doc_ids = shuffle(pivot_ids, target_ids, doc_ids, random_state=0)

    return (idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids)

