import pandas as pd
from preprocess import DataProcessor
import numpy as np
from data_utils import load_preprocessed_data

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

print(len(word_to_idx))
print(len(idx_to_word))