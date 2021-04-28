import tensorflow as tf

class WordEmbedding:
  
    def __init__(self, embedding_size, vocab_size, sample_size, power=0.75, freqs=None):
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.power = power
        self.freqs = freqs

        self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1.0, 1.0),
                           name="word_embedding")

        self.nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                           stddev=tf.sqrt(1 / embedding_size)),
                                                           name="nce_weights")
        self.nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")

    def compute_loss(self, embed, train_labels):
        with tf.name_scope("negative_sampling"):

            train_labels = tf.reshape(train_labels, [tf.shape(train_labels)[0], 1])

            sampler = tf.nn.fixed_unigram_candidate_sampler(train_labels,
                                                            num_true=1,
                                                            num_sampled=self.sample_size,
                                                            unique=True,
                                                            range_max=self.vocab_size,
                                                            distortion=self.power,
                                                            unigrams=self.freqs)

            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=self.sample_size,
                               num_classes=self.vocab_size,
                               num_true=1,
                               sampled_values=sampler))
        return loss
