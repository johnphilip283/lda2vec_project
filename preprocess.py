from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from spacy.lang.en import English
import pandas as pd
import numpy as np
import spacy
import pickle
import os

class DataProcessor:

    def __init__(self, df, textcol, max_features=30000, window_size=5):
        self.df = df
        self.textcol = textcol
        self.disallowed = ("ax>", '`@("', '---', '===', '^^^', "AX>", "GIZ")
        self.max_features = max_features
        self.window_size = window_size
        self.nlp = English()
        self.nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

    def clean(self, line):
        return ' '.join(word for word in line.split() if not any(term in word for term in self.disallowed))

    def preprocess(self, path_to_save):
        texts = [str(self.clean(text)) for text in self.df[self.textcol].values.tolist()]
        texts_clean = []

        for doc in self.nlp.pipe(texts, disable=['ner', 'parser']):          
            texts_clean.append(" ".join([token.lower_ for token in doc if token.is_alpha and not token.is_stop]))

        tokenizer = Tokenizer(self.max_features, filters="", lower=False)
        tokenizer.fit_on_texts(texts_clean)
        tokenizer.word_index["<UNK>"] = 0
        tokenizer.word_docs["<UNK>"] = 0
        tokenizer.word_counts["<UNK>"] = 0
        idx_data = tokenizer.texts_to_sequences(texts_clean)
        word_to_idx = tokenizer.word_index
        idx_to_word = {v: k for k, v in word_to_idx.items()}

        vocab_size = min(self.max_features, len(idx_to_word))

        freqs = [tokenizer.word_counts[idx_to_word[i]] for i in range(vocab_size)]

        skipgram_data = []
        excluded_docs = []
        doc_lengths = []

        doc_id_counter = 0
       
        for i, text in enumerate(idx_data):
            pairs, _ = skipgrams(text, vocabulary_size=vocab_size, window_size=self.window_size, shuffle=True, negative_samples=0)
            if len(pairs) > 2:
                for pair in pairs:
                    temp_data = pair
                    temp_data.append(doc_id_counter)
                    temp_data.append(i)
                    skipgram_data.append(temp_data)
                doc_lengths.append(len(text))
                doc_id_counter += 1
            else:
                excluded_docs.append(i)

        skipgrams_df = pd.DataFrame(skipgram_data)

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        # Save vocabulary dictionaries to file
        with open(f"{path_to_save}/idx_to_word.pickle", "wb") as idx_to_word_out:
          pickle.dump(idx_to_word, idx_to_word_out)
        
        with open(f"{path_to_save}/word_to_idx.pickle", "wb") as word_to_idx_out:
          pickle.dump(word_to_idx, word_to_idx_out)
          
        np.save(f"{path_to_save}/doc_lengths", doc_lengths)
        np.save(f"{path_to_save}/freqs", freqs)

        skipgrams_df.to_csv(f"{path_to_save}/skipgrams.txt", sep="\t", index=False, header=None)