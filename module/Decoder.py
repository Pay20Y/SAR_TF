import tensorflow as tf

class Decoder(object):
    def __init__(self, output_classes, hidden_nums=512, layer_nums=2, embedding_dim=512 ,seq_len=40, lstm_keep_prob=1.0, att_keep_prob=0.5, is_training=True):
        self.hidden_nums = hidden_nums
        self.layer_nums = layer_nums
        self.output_classes = output_classes
        self.is_training = is_training
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.is_training = is_training
        if self.is_training:
            self.lstm_keep_prob = lstm_keep_prob
            self.att_keep_prob = att_keep_prob
        else:
            self.lstm_keep_prob = 1.0
            self.att_keep_prob = 1.0
        self.START_TOKEN = output_classes - 3 # Same like EOS TOKEN

        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

    def _word_embedding(self, inputs, reuse=False):
        """
        Embedding the input character
        :param inputs: [N * self.output_classes] one-hot tensor
        :param reuse:
        :return: N * self.embedding_dim
        """
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('embed_w', [self.output_classes + 1, self.embedding_dim], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def mask_softmax(self, input_tensor, mask):
        """
        Fill input_tensor not in mask to a inf
        :param input_tensor: N * ~
        :param mask: N * ~
        :return: N * ~
        """
        mask = mask * (-9999)
        input_tensor = input_tensor + mask
        output_tensor = tf.nn.softmax(input_tensor, axis=-1)
        return output_tensor

    def attention_op(self,hidden_state, feature_map, mask, reuse=False):
        """
        2D attention on feature map
        :param hidden_state: N * 512
        :param feature_map: N * H * W * 512
        :param mask: N * H * W
        :param reuse:
        :return:
        """
        with tf.variable_scope("Attention", reuse=reuse):
            feamap_filter = tf.get_variable(name='feamap_filter',
                                            shape=[3, 3, self.hidden_nums, self.embedding_dim],
                                            initializer=self.weight_initializer)
            feamap_bias = tf.get_variable(name='feamap_bias',
                                          shape=[self.embedding_dim],
                                          initializer=self.const_initializer)
            state_weights = tf.get_variable(name='state_weights',
                                            shape=[self.hidden_nums, self.embedding_dim],
                                            initializer=self.weight_initializer)
            state_bias = tf.get_variable(name='state_bias',
                                         shape=[self.embedding_dim],
                                         initializer=self.const_initializer)
            attention_weights = tf.get_variable(name='attention_weights',
                                                shape=[self.embedding_dim, 1],
                                                initializer=self.weight_initializer)
            attention_bias = tf.get_variable(name='attention_bias',
                                             shape=[1],
                                             initializer=self.const_initializer)

            N = tf.shape(feature_map)[0]
            H = tf.shape(feature_map)[1]
            W = tf.shape(feature_map)[2]

            # Convolution on feature map
            neighbor_feature_map = tf.nn.conv2d(feature_map, feamap_filter, strides=[1, 1, 1, 1], padding='SAME')
            neighbor_feature_map = tf.nn.bias_add(neighbor_feature_map, feamap_bias)
            neighbor_feature_map = tf.reshape(neighbor_feature_map, [N, -1, self.embedding_dim]) # N * (H * W) * C

            # Linear on hidden state
            hidden_state = tf.add(tf.matmul(hidden_state, state_weights), state_bias)
            hidden_state = tf.tile(tf.expand_dims(hidden_state, axis=1), [1, tf.shape(neighbor_feature_map)[1], 1])

            # Fusing feature map and hidden state
            fusion_feature = tf.nn.dropout(tf.nn.tanh(tf.add(neighbor_feature_map, hidden_state)), keep_prob=self.att_keep_prob)
            fusion_feature = tf.reshape(fusion_feature, [-1, self.embedding_dim])

            attention_logits = tf.add(tf.matmul(fusion_feature, attention_weights), attention_bias)
            mask = tf.reshape(mask, [N, -1])
            alpha = self.mask_softmax(tf.reshape(attention_logits, [-1, tf.shape(neighbor_feature_map)[1]]), mask)
            feature_map = tf.reshape(feature_map, [N, -1, self.embedding_dim])
            glimpse = tf.matmul(tf.expand_dims(alpha, axis=1), feature_map) # N * 1 * 512
            glimpse = tf.squeeze(glimpse, axis=1)

            alpha = tf.reshape(alpha, [-1, H, W])
            return glimpse, alpha

    def decode_op(self, glimpse, hidden_state, reuse=False):
        with tf.variable_scope("logits", reuse=reuse):
            output_w = tf.get_variable(name="output_w",
                                       shape=[2 * self.hidden_nums, self.output_classes],
                                       initializer=self.weight_initializer)
            output_b = tf.get_variable(name="output_b",
                                       shape=[self.output_classes],
                                       initializer=self.const_initializer)

            logistic = tf.matmul(tf.concat([glimpse, hidden_state], axis=1), output_w) + output_b

            return logistic

    def __call__(self, encoder_state, feature_map, labels, mask):
        with tf.variable_scope("Decoder"):
            LSTM_cell = [tf.nn.rnn_cell.LSTMCell(num_units=n, name="deocde_lstm_cell_{}".format(i)) for i, n in enumerate([self.hidden_nums] * self.layer_nums)]
            LSTM_module = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(LSTM_cell), input_keep_prob=self.lstm_keep_prob, output_keep_prob=self.lstm_keep_prob)

            states = LSTM_module.zero_state(tf.shape(encoder_state)[0], tf.float32)
            labels = tf.split(labels, self.seq_len, axis=1)
            outputs = []
            attention_weights = []
            for t in range(self.seq_len + 1):
                if t == 0:
                    inputs_y = encoder_state
                elif t == 1:
                    inputs_y = tf.fill([tf.shape(feature_map)[0]], value=self.START_TOKEN)
                    inputs_y = self._word_embedding(inputs_y, reuse=(t != 1))
                else:
                    if self.is_training:
                        inputs_y = tf.squeeze(labels[t-2], axis=1)
                    else:
                        inputs_y = tf.cast(tf.argmax(outputs[t-1], axis=-1), tf.int32)
                    inputs_y = self._word_embedding(inputs_y, reuse=(t != 1))

                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, states = LSTM_module(inputs_y, state=states)

                h = states[-1].h
                glimpse, att_weight = self.attention_op(h, feature_map, mask, reuse=(t!=0))

                logistic = self.decode_op(glimpse, h, reuse=(t!=0))

                outputs.append(logistic)
                attention_weights.append(att_weight)

            outputs = outputs[1:]
            attention_weights = attention_weights[1:]

            outputs = tf.transpose(tf.stack(outputs, axis=0), [1, 0, 2])
            attention_weights = tf.transpose(tf.stack(attention_weights, axis=0), [1, 0, 2, 3])

            return outputs, attention_weights

    def beam_search(self, encoder_state, feature_map, mask, beam_width=5):
        N, H, W, C = feature_map.shape.as_list()
        N = N if N is not None else tf.shape(feature_map)[0]
        H = H if H is not None else tf.shape(feature_map)[1]
        W = W if W is not None else tf.shape(feature_map)[2]
        C = C if C is not None else tf.shape(feature_map)[3]

        assert N == 1, "beam search only support for test with batch size 1"

        with tf.variable_scope("Decoder"): # Since only using beam search in evaluating
            with tf.variable_scope("lstm"):
                LSTM_cell = [tf.nn.rnn_cell.LSTMCell(num_units=n, name="deocde_lstm_cell_{}".format(i)) for i, n in enumerate([self.hidden_nums] * self.layer_nums)]
                LSTM_module = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(LSTM_cell), input_keep_prob=self.lstm_keep_prob, output_keep_prob=self.lstm_keep_prob)

            # inflated_encoder_state = tf.reshape(tf.transpose(tf.tile(tf.transpose(tf.expand_dims(encoder_state, axis=1), [1, 0, 2]), [beam_width, 1, 1]), [1, 0, 2]), [-1, encoder_state.shape[-1]])
            # inflated_feature_map = tf.reshape(tf.transpose(tf.tile(tf.transpose(tf.expand_dims(feature_map, axis=1), [1, 0, 2, 3, 4]), [beam_width, 1, 1, 1, 1]), [1, 0, 2, 3, 4]), [-1, H, W, C])
            # inflated_mask_map = tf.reshape(tf.transpose(tf.tile(tf.transpose(tf.expand_dims(mask, axis=1), [1, 0, 2, 3, 4]), [beam_width, 1, 1, 1, 1]), [1, 0, 2, 3, 4]), [-1, H, W, 1])

            states = LSTM_module.zero_state(N, tf.float32) # 1 * 512
            with tf.variable_scope('lstm', reuse=False):
                _, states = LSTM_module(encoder_state, state=states)

            # states = tf.tile(states, [beam_width, 1]) # B * 512
            # Multi-layer LSTM cell can not tile directly can it?
            states_ = []
            for state in states:
                states_.append(tf.nn.rnn_cell.LSTMStateTuple(tf.tile(state.c, [beam_width, 1]), tf.tile(state.h, [beam_width, 1])))
            states = tuple(states_)

            feature_map = tf.tile(feature_map, [beam_width, 1, 1, 1]) # B * H * W * C
            mask_map = tf.tile(mask, [beam_width, 1, 1, 1])

            # sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)]) # 1 * B
            sel_sum_logprobs = tf.log([1.] + [0.] * (beam_width - 1)) # B

            # beam_mask = tf.ones([1, beam_width]) # 1 * B
            beam_mask = tf.ones([beam_width]) # B

            # ids = tf.tile([[self.START_TOKEN]], [1, beam_width]) # 1 * B
            ids = tf.tile([self.START_TOKEN], [beam_width]) # B
            # sel_ids = tf.zeros([1, beam_width, 0], dtype=ids.dtype) # 1 * B * 0
            sel_ids = tf.zeros([beam_width, 0], dtype=ids.dtype) # B * 0

            attention_weights = []

            for t in range(self.seq_len + 1):
                inputs_y = self._word_embedding(tf.reshape(ids, [-1]), reuse=(t != 0))
                with tf.variable_scope('lstm', reuse=True):
                    _, states = LSTM_module(inputs_y, state=states)
                h = states[-1].h
                glimpse, att_weight = self.attention_op(h, feature_map, mask_map, reuse=(t != 0))
                attention_weights.append(att_weight)
                logistic = self.decode_op(glimpse, h, reuse=(t != 0))
                # logistic = tf.reshape(tf.nn.log_softmax(logistic), [1, beam_width, self.output_classes]) # 1 * B * C
                logistic = tf.nn.log_softmax(logistic) # B * C

                sum_logprobs = (tf.expand_dims(sel_sum_logprobs, axis=1) + (logistic * tf.expand_dims(beam_mask, axis=1))) # B * C

                sel_sum_logprobs, indices = tf.nn.top_k(tf.reshape(sum_logprobs, [self.output_classes * beam_width]), k=beam_width) # B

                ids = indices % self.output_classes # B
                beam_ids = indices // self.output_classes # B

                # states = [(c_, h_) for state in states for c_ in tf.gather(state.c, beam_ids) for h_ in tf.gather(state.h, beam_ids) ]
                states = tuple([(tf.gather(state.c, beam_ids), tf.gather(state.h, beam_ids)) for state in states])
                sel_ids = tf.concat([tf.gather(sel_ids, beam_ids), tf.expand_dims(ids, axis=1)], axis=1)
                beam_mask = (tf.gather(beam_mask, beam_ids) * tf.to_float(tf.not_equal(ids, self.START_TOKEN)))

            # return (sel_ids, sel_sum_logprobs), attention_weights
            return sel_ids[0], attention_weights

if __name__ == "__main__":
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    decoder = Decoder(output_classes=95, is_training=False)
    # attention_unit = AttentionUnit(512)
    input_feature_map = tf.placeholder(dtype=tf.float32, shape=[1, 6, 40, 512])
    input_encoder_state = tf.placeholder(dtype=tf.float32, shape=[1, 512])
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, 40])
    input_masks = tf.placeholder(dtype=tf.float32, shape=[1, 6, 40, 1])
    outputs = decoder(input_encoder_state, input_feature_map, input_labels, input_masks)
    # outputs = decoder.beam_search(input_encoder_state, input_feature_map, input_masks)
    input_labels_one_hot = tf.one_hot(input_labels, 95, 1, 0)  # N * L * C
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels_one_hot, logits=tf.cast(input_labels_one_hot, tf.float32), dim=-1))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels_one_hot, logits=outputs[0], dim=-1))
    optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
    """
    input_labels = []
    for i in range(40):
        input_labels.append(tf.placeholder(dtype=tf.int32, shape=[None]))
    """

    _input_feature_map = np.random.rand(1, 6, 40, 512)
    _input_encoder_state = np.random.rand(1, 512)
    _input_labels = np.random.rand(1, 40)
    _input_mask = np.random.rand(1, 6, 40, 1)
    with tf.Session() as sess:
        for i in range(10):
            sess.run(tf.global_variables_initializer())

            _, loss_value, decoder_value, input_labels_one_hot_value = sess.run([optimizer, loss, outputs, input_labels_one_hot],
                                                    feed_dict={input_feature_map: _input_feature_map,
                                                               input_encoder_state: _input_encoder_state,
                                                               input_labels: _input_labels,
                                                               input_masks: _input_mask})
            """
            loss_value, decoder_value = sess.run([loss, outputs],
                                                 feed_dict={input_feature_map: _input_feature_map,
                                                            input_encoder_state: _input_encoder_state,
                                                            input_labels: _input_labels,
                                                            input_masks: _input_mask})
            """
            # print(loss_value)
            print(input_labels_one_hot_value)
