import tensorflow as tf

class Encoder(object):
    def __init__(self, hidden_nums=512, layer_nums=2, keep_prob=1.0, is_training=True):

        self.hidden_nums = hidden_nums
        self.layer_nums = layer_nums
        self.is_training = is_training
        if self.is_training:
            self.keep_prob = keep_prob
        else:
            self.keep_prob = 1.0
    def __call__(self, feature_map, input_widths):
        """
		:param feature_map: [N, H, W, C] feature map after backbone
		:param input_widths: [N] indicating valid length of feature map
		:return: [N, hidden_num] final hidden state of the second LSTM layer
		"""
        with tf.variable_scope("Encoder"):
            # Vertical max pooling
            pool_feature_map = tf.reduce_max(feature_map, axis=1, keepdims=False) # B * W * C

            # 2 layer LSTM
            LSTM_cell = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in [self.hidden_nums] * self.layer_nums]
            LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(LSTM_cell), input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(LSTM_cell, pool_feature_map, sequence_length=input_widths, dtype=tf.float32)
            print("Encoder LSTM state: ", state[-1])

        return state[-1].h

if __name__ == '__main__':
    ec = Encoder()
    input_fea_map = tf.placeholder(tf.float32, shape=[16, None ,64, 512], name="input_fea_map")
    input_encoder_label = tf.placeholder(tf.float32, shape=[16, 512], name="input_encoder_label")
    encoder_state = ec(input_fea_map)
    loss = tf.reduce_mean(tf.abs(input_encoder_label - encoder_state))
    optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
    import numpy as np
    _input_fea_map = np.random.rand(16, 8, 64, 512)
    _input_encoder_label = np.random.rand(16, 512)
    with tf.Session() as sess:
        for i in range(10):
            sess.run(tf.global_variables_initializer())
            _, loss_value = sess.run([optimizer, loss], feed_dict={input_fea_map: _input_fea_map, input_encoder_label: _input_encoder_label})
            print(loss_value)
    # print("encoder_state: ", encoder_state)