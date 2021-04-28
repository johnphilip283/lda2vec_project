import numpy as np
import pyLDAvis
import random

def softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def generate_ldavis_data(clean_data_dir, model, idx_to_word, freqs, vocab_size):

    doc_embed = model.sesh.run(model.mixture.doc_embedding)
    topic_embed = model.sesh.run(model.mixture.topic_embedding)
    word_embed = model.sesh.run(model.w_embed.embedding)

    vocabulary = []
    for _, v in idx_to_word.items():
        vocabulary.append(v)

    doc_lengths = np.load(clean_data_dir + "/doc_lengths.npy")

    vis_data = prepare_topics(doc_embed, topic_embed, word_embed, np.array(vocabulary), doc_lengths=doc_lengths,
                              term_frequency=freqs)

    prepared_vis_data = pyLDAvis.prepare(**vis_data)
    pyLDAvis.show(prepared_vis_data)

def prepare_topics(weights, factors, word_vectors, vocab, doc_lengths=None, term_frequency=None):

    topic_to_word = []
    msg = "Vocabulary size did not match size of word vectors"
    assert len(vocab) == word_vectors.shape[0], msg
    word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]
    for factor_vector in factors:
        dot = np.dot(word_vectors, factor_vector)
        e_x = np.exp(dot - np.max(dot))
        factor_to_word = e_x / e_x.sum()
        topic_to_word.append(np.ravel(factor_to_word))

    topic_to_word = np.array(topic_to_word)
    msg = "Not all rows in topic_to_word sum to 1"
    assert np.allclose(np.sum(topic_to_word, axis=1), 1), msg

    doc_to_topic = softmax_2d(weights)
    msg = "Not all rows in doc_to_topic sum to 1"
    assert np.allclose(np.sum(doc_to_topic, axis=1), 1), msg
    data = {'topic_term_dists': topic_to_word,
            'doc_topic_dists': doc_to_topic,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}
    return data

def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    keypoints = []
    for i in range(0, len(args[0]), n):
        keypoints.append((i, i + n))
    random.shuffle(keypoints)
    for a, b in keypoints:
        yield [arg[a: b] for arg in args]