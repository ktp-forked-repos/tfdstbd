from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .dataset import Trainer


trainer_params = {
    'data_dir': 'data',
    'batch_size': 20,
    'test_size': 0.2,
    'doc_size': 10,
    'random_seed': 43,
}
trainer = Trainer(**trainer_params)

# for z in trainer._test_generator():
#     print(z)


# from collections import Counter
# from builtins import range
# from os import path
# import tensorflow as tf
# from .ops import split_tokens
#
# with open(path.join(path.dirname(__file__), 'data', 'sentences.txt'), 'rb') as fin:
#     data = fin.read().decode('utf-8').split('\n\n')
#
# batch_ph = tf.placeholder(tf.string, shape=(None,))
# tokens_op = split_tokens(batch_ph)
# vocab = Counter()
# with tf.Session() as sess:
#     for x in range(0, len(data), 1000):
#         tokens = sess.run(tokens_op, feed_dict={batch_ph: data[x:x + 1000]})
#         vocab.update(tokens.values)
#
# with open('vocab.txt', 'wb') as fout:
#     for w, f in vocab.most_common():
#         if len(w) > 2:
#             continue
#         line = u'{}\t{}\n'.format(w.decode('utf-8'), f)
#         fout.write(line.encode('utf-8'))
