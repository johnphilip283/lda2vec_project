import pandas as pd
from preprocess import DataProcessor
import numpy as np
from data_utils import load_preprocessed_data
from model import LDA2Vec
from visualization import generate_ldavis_data

data_dir = "data"

clean_data_dir = f"{data_dir}/clean_data"

input_file = "20_newsgroups.txt"

df = pd.read_csv(f"{data_dir}/{input_file}", sep="\t")

processor = DataProcessor(df, "texts", max_features=10000)
processor.preprocess(clean_data_dir)

(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids) = load_preprocessed_data(clean_data_dir)

num_docs = len(np.unique(doc_ids))
vocab_size = len(idx_to_word)

params = {
        "freqs": freqs, 
        "batch_size": 512,
        "save_graph": True
        }

embed_size = 128
num_topics = 20
num_epochs = 200


model = LDA2Vec(num_docs,
          vocab_size,
          num_topics,
          embed_size,
          **params)

model.train(pivot_ids,
        target_ids,
        doc_ids,
        num_epochs,
        idx_to_word)

generate_ldavis_data(clean_data_dir, model, idx_to_word, freqs, vocab_size)