import tensorflow as tf
import numpy as np

class EmbeddingMixture:

    def __init__(self, n_documents, n_topics, n_dim):
        self.n_documents = n_documents
        scalar = 1 / np.sqrt(n_documents + n_topics)
        
        self.doc_embedding = tf.Variable(tf.random_normal([n_documents, n_topics], mean=0, stddev=50 * scalar),
                                         name='doc_embedding')

        self.topic_embedding = tf.get_variable('topic_embedding', shape=[n_topics, n_dim],
                                               dtype=tf.float32,
                                               initializer=tf.orthogonal_initializer(gain=scalar))

    def get_context(self, doc_ids):
      w = tf.nn.embedding_lookup(self.doc_embedding, doc_ids, name='doc_proportions')
      proportions = tf.nn.softmax(w)
      return tf.matmul(proportions, self.topic_embedding, name='docs_mul_topics')
