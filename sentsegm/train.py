from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __package__ is None:
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from data_loader import TrainDataLoader
else:
    from rusentseg.lib import TrainDataLoader

data_params = {
    'data_dir': 'data/',
    'batch_size': 32,
    'test_size': 0.2,
    'max_docsize': 10,
}
data_loader = TrainDataLoader(data_params['data_dir'],
                              batch_size=data_params['batch_size'],
                              test_size=data_params['test_size'],
                              max_docsize=data_params['max_docsize'])

# char_rnn = CharRNN()
# model_params = {
#     'vocab_size': data_loader.vocab_size(),
#     'rnn_size': 512,
#     'num_layers': 3,
#     'input_keep_prob': 0.8,
#     'output_keep_prob': 0.8,
#     'batch_size': data_params['batch_size'],
#     'seq_length': data_params['seq_length'],
#     # 'grad_clip': 5.0,
#     # 'log_dir': 'logs',
#     'learning_rate': 0.001,
# }
# estimator = tf.estimator.Estimator(
#     model_fn=char_rnn.model_fn,
#     model_dir='rusentseg',
#     params=model_params)
#
# train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()
#
# train_input_fn, train_input_hook = dataset.get_train_inputs(train_X, train_y)
# test_input_fn, test_input_hook = dataset.get_test_inputs(test_X, test_y)
#
# estimator.train(input_fn=train_input_fn, steps=10000)


X, y = next(data_loader._test_generator())
for p in zip(X, y):
    print(p)
