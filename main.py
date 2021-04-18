import pandas as pd
from preprocess import DataProcessor
import numpy as np
from data_utils import load_preprocessed_data
from model import LDA2Vec

# Data directory
data_dir ="data"
# Where to save preprocessed data
clean_data_dir = f"{data_dir}/clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"

# Read in data file
df = pd.read_csv(f"{data_dir}/{input_file}", sep="\t")

# Initialize a preprocessor
processor = DataProcessor(df, "texts", max_features=10000)
processor.preprocess(clean_data_dir)

(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids) = load_preprocessed_data(clean_data_dir)

num_docs = len(np.unique(doc_ids))
vocab_size = len(idx_to_word)

embed_size = 128
num_topics = 20
switch_loss_epoch = 5

save_graph = True
num_epochs = 200
batch_size = 512

# Initialize the model
model = LDA2Vec(num_docs,
          vocab_size,
          num_topics,
          embedding_size=embed_size,
          freqs=freqs,
          batch_size=batch_size,
          save_graph_def=save_graph)

# Train the model
model.train(pivot_ids,
        target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        idx_to_word=idx_to_word,
        switch_loss_epoch=switch_loss_epoch)

# Visualize topics with pyldavis
generate_ldavis_data(clean_data_dir, model, idx_to_word, freqs, vocab_size)