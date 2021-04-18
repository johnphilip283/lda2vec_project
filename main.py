import pandas as pd
from preprocess import DataProcessor

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