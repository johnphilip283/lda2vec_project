from word_embedding import WordEmbedding
from embedding_mixture import EmbeddingMixture
from datetime import datetime
import tensorflow as tf
import numpy as np
from data_utils import chunks

class LDA2Vec:
  
  def __init__(self, num_docs, vocab_size, num_topics, embedding_size, freqs, batch_size, save_graph, num_sampled=40):
    self.num_docs = num_docs
    self.vocab_size = vocab_size
    self.num_topics = num_topics
    self.embedding_size = embedding_size
    self.freqs = freqs
    self.batch_size = batch_size
    self.save_graph = save_graph
    self.num_sampled = num_sampled
    self.lmbda = 200.0
    self.learning_rate = 0.001
    self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.sesh = tf.Session(config=self.config)
    self.computed_norm = False

    self.logdir = "_".join(("lda2vec", datetime.now().strftime('%y%m%d_%H%M')))

    self.w_embed = WordEmbedding(self.embedding_size, self.vocab_size, self.num_sampled, freqs=self.freqs)

    self.mixture = EmbeddingMixture(self.num_docs, self.num_topics, self.embedding_size)

    handles = self.retrieve_variables()

    (self.x, self.y, self.docs, self.step, self.switch_loss,
    self.word_context, self.doc_context, self.loss_word2vec,
    self.fraction, self.loss_lda, self.loss, self.loss_avgs_op,
    self.optimizer, self.merged) = handles

  def train(self, pivot_ids, target_ids, doc_ids, num_epochs, idx_to_word, switch_loss_epoch=5, save_every=1, report_every=1, print_topics_every=5):
    data_size = len(pivot_ids)

    temp_fraction = self.batch_size * 1.0 / data_size

    self.sesh.run(tf.assign(self.fraction, temp_fraction))

    iters_per_epoch = int(data_size / self.batch_size) + np.ceil(data_size % self.batch_size)

    switch_loss_step = iters_per_epoch * switch_loss_epoch

    self.sesh.run(tf.assign(self.switch_loss, switch_loss_step))

    if self.save_graph:

        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(self.logdir + '/', graph=self.sesh.graph)

    for epoch in range(num_epochs):
        print('\nEPOCH:', epoch + 1)

        for pivot, target, doc in chunks(self.batch_size, pivot_ids, target_ids, doc_ids):
            
            feed_dict = {self.x: pivot, self.y: target, self.docs: doc}
            
            fetches = [self.merged, self.optimizer, self.loss,
                        self.loss_word2vec, self.loss_lda, self.step]
            
            summary, _, l, lw2v, llda, step = self.sesh.run(fetches, feed_dict=feed_dict)

        if (epoch + 1) % report_every == 0:
            print('Loss: ', l, 'Word2Vec Loss: ', lw2v, 'LDA loss: ', llda)

        if (epoch + 1) % save_every == 0 and self.save_graph:
            writer.add_summary(summary, step)
            writer.flush()
            writer.close()
            save_path = saver.save(self.sesh, self.logdir + '/model.ckpt')
            writer = tf.summary.FileWriter(self.logdir + '/', graph=self.sesh.graph)
        
        if epoch > 0 and (epoch + 1) % print_topics_every == 0:
            idxs = np.arange(self.num_topics)
            words, sims = self.get_k_closest(idxs, idx_to_word=idx_to_word, k=10)

    if self.save_graph and (epoch + 1) % save_every != 0:
        writer.add_summary(summary, step)
        writer.flush()
        writer.close()        
        save_path = saver.save(self.sesh, self.logdir + '/model.ckpt')

  def get_k_closest(self, idxs, in_type="topic", vs_type="word", k=10, idx_to_word=None):
    if not self.computed_norm:
      self.normed_embed_dict = {}
      norm = tf.sqrt(tf.reduce_sum(self.mixture.topic_embedding ** 2, 1, keep_dims=True))
      self.normed_embed_dict['topic'] = self.mixture.topic_embedding / norm
      norm = tf.sqrt(tf.reduce_sum(self.w_embed.embedding ** 2, 1, keep_dims=True))
      self.normed_embed_dict['word'] = self.w_embed.embedding / norm
      norm = tf.sqrt(tf.reduce_sum(self.mixture.doc_embedding ** 2, 1, keep_dims=True))
      self.normed_embed_dict['doc'] = self.mixture.doc_embedding / norm
      self.idxs_in = tf.placeholder(tf.int32, shape=[None], name='idxs')
      self.computed_norm = True

    self.batch_array = tf.nn.embedding_lookup(self.normed_embed_dict[in_type], self.idxs_in)
    self.cosine_similarity = tf.matmul(self.batch_array, tf.transpose(self.normed_embed_dict[vs_type], [1, 0]))
    feed_dict = {self.idxs_in: idxs}
    sim, sim_idxs = self.sesh.run(tf.nn.top_k(self.cosine_similarity, k=k), feed_dict=feed_dict)
    if idx_to_word:
      print('---------Closest {} words to given indexes----------'.format(k))
      for i, idx in enumerate(idxs):
        in_word = 'Topic ' + str(idx)
        vs_word_list = []
        for vs_i in range(sim_idxs[i].shape[0]):
            vs_idx = sim_idxs[i][vs_i]
            vs_word = idx_to_word[vs_idx]
            vs_word_list.append(vs_word)
            print(in_word, ':', (', ').join(vs_word_list))

    return (sim, sim_idxs)

  def retrieve_variables(self):
    x = tf.placeholder(tf.int32, shape=[None], name='x_pivot_idxs')
    y = tf.placeholder(tf.int64, shape=[None], name='y_target_idxs')
    docs = tf.placeholder(tf.int32, shape=[None], name='doc_ids')

    step = tf.Variable(0, trainable=False, name='global_step')

    switch_loss = tf.Variable(0, trainable=False)
    word_context = tf.nn.embedding_lookup(self.w_embed.embedding, x, name='word_embed_lookup')
    doc_context = self.mixture.get_context(doc_ids=docs)

    contexts_to_add = [word_context, doc_context]
    context = tf.add_n(contexts_to_add, name='context_vector')

    with tf.name_scope('nce_loss'):
        loss_word2vec = self.w_embed.compute_loss(context, y)
        tf.summary.scalar('nce_loss', loss_word2vec)

    with tf.name_scope('lda_loss'):
        fraction = tf.Variable(1, trainable=False, dtype=tf.float32, name='fraction')
        loss_lda = self.lmbda * fraction * self.prior()
        tf.summary.scalar('lda_loss', loss_lda)

    loss = tf.cond(step < switch_loss, lambda: loss_word2vec, lambda: loss_word2vec + loss_lda)

    loss_avgs_op = self.moving_avgs.apply([loss_lda, loss_word2vec, loss])
    
    with tf.control_dependencies([loss_avgs_op]):
        optimizer = tf.contrib.layers.optimize_loss(loss,
                                                    tf.train.get_global_step(),
                                                    self.learning_rate,
                                                    'Adam',
                                                    name='Optimizer')
    
    self.sesh.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
    
    merged = tf.summary.merge_all()

    return [x, y, docs, step, switch_loss, word_context, doc_context,
                  loss_word2vec, fraction, loss_lda, loss, loss_avgs_op, optimizer, merged]

  def prior(self):
    n_topics = self.mixture.doc_embedding.get_shape()[1].value
    alpha = 1.0 / n_topics
    log_proportions = tf.nn.log_softmax(self.mixture.doc_embedding)
    return tf.reduce_sum((alpha - 1.0) * log_proportions) 