import tensorflow as tf
from module.Backbone import Backbone
from module.Encoder import Encoder
from module.Decoder import Decoder
from tensorflow.contrib import slim
class SARModel(object):
    def __init__(self,
                 num_classes,
                 encoder_dim=512,
                 encoder_layer=2,
                 decoder_dim=512,
                 decoder_layer=2,
                 decoder_embed_dim=512,
                 seq_len=40,
                 beam_width=5,
                 is_training=True):
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.encoder_layer = encoder_layer
        self.decoder_dim = decoder_dim
        self.decoder_layer = decoder_layer
        self.decoder_embed_dim = decoder_embed_dim
        self.seq_len = seq_len
        self.beam_width = beam_width
        self.is_training = is_training

        self.backbone = Backbone(is_training=self.is_training)
        self.encoder = Encoder(hidden_nums=self.encoder_dim, layer_nums=self.encoder_layer, is_training=self.is_training)
        self.decoder = Decoder(output_classes=self.num_classes,
                               hidden_nums=self.decoder_dim,
                               layer_nums=self.decoder_layer,
                               embedding_dim=self.decoder_embed_dim ,
                               seq_len=self.seq_len,
                               lstm_keep_prob=1.0,
                               att_keep_prob=0.5,
                               is_training=self.is_training)

    def __call__(self, input_images, input_labels, input_widths, batch_size, reuse=False, decode_type='greed'):
        with tf.variable_scope(name_or_scope="sar", reuse=reuse):
            encoder_state,  feature_map, mask_map = self.inference(input_images, input_widths, batch_size)
            decoder_logits, attention_weights, pred = self.decode(encoder_state, feature_map, input_labels, mask_map, decode_type=decode_type)

            return decoder_logits, attention_weights, pred

    def inference(self, input_images, input_widths, batch_size):
        # with tf.variable_scope(name_or_scope='sar', reuse=reuse):
        img_W = tf.cast(tf.shape(input_images)[2], tf.float32)
        feature_map = self.backbone(input_images)
        fea_W = tf.cast(tf.shape(feature_map)[2], tf.float32)
        input_widths = tf.cast(tf.math.floor(input_widths * (fea_W / img_W)), tf.int32)
        encoder_state = self.encoder(feature_map, input_widths)

        with tf.name_scope(name="fea_post_process"):
            # construct mask map
            input_widths_list = tf.split(input_widths, batch_size)
            mask_map = []
            for i, width in enumerate(input_widths_list):
                mask_slice = tf.pad(tf.zeros(dtype=tf.float32, shape=width), [[0, tf.shape(feature_map)[2]-width[0]]], constant_values=1)
                mask_slice = tf.tile(tf.expand_dims(mask_slice, axis=0), [tf.shape(feature_map)[1], 1])
                mask_map.append(mask_slice)
            # mask_map = tf.expand_dims(tf.zeros_like(feature_map[:, :, :, 0]), axis=-1)  # N * H * W * 1
            mask_map = tf.stack(mask_map, axis=0)
            mask_map = tf.expand_dims(mask_map, axis=3) # N * H * W * 1
            reverse_mask_map = 1 - mask_map
            feature_map = feature_map * reverse_mask_map

        return encoder_state, feature_map, mask_map

    def loss(self, pred, input_labels, input_lengths_mask):
        """
        cross-entropy loss
        :param pred: Decoder outputs N * L * C
        :param input_labels: N * L
        :param input_lengths_mask: N * L (0 & 1 like indicating the real length of the label)
        :return:
        """
        with tf.name_scope(name="MaskCrossEntropyLoss"):
            input_labels = tf.one_hot(input_labels, self.num_classes, 1, 0) # N * L * C
            input_labels = tf.stop_gradient(input_labels) # since softmax_cross_entropy_with_logits_v2 will bp to labels
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels, logits=pred, dim=-1)
            mask_loss = loss * tf.cast(input_lengths_mask, tf.float32)

            loss = tf.reduce_sum(mask_loss) / tf.cast(tf.shape(pred)[0], tf.float32)

            return loss

    def decode(self, encoder_state, feature_map, input_labels, mask_map, decode_type='greed'):
        assert decode_type.lower() in ['greed', 'beam', 'lexicon'], "Not support decode type"
        # with tf.variable_scope(name_or_scope='sar', reuse=reuse):
        if decode_type.lower() == "greed":
            decoder_outputs, attention_weights = self.decoder(encoder_state, feature_map, input_labels, mask_map)
            pred = tf.argmax(decoder_outputs, axis=2)
            return decoder_outputs, attention_weights, pred
        elif decode_type == "beam" and not self.is_training:
            pred, attention_weights = self.decoder.beam_search(encoder_state, feature_map, mask_map, self.beam_width)
            return None , attention_weights, pred
        elif decode_type == "lexicon":
            return None

def test():
    input_images = tf.placeholder(dtype=tf.float32, shape=[32, 48, 160, 3])
    input_labels = tf.placeholder(dtype=tf.int32, shape=[32, 40])
    input_lengths = tf.placeholder(dtype=tf.int32, shape=[32, 40])
    input_feature_map = tf.placeholder(dtype=tf.float32, shape=[32, 12, 20, 512])
    input_widths = tf.placeholder(dtype=tf.float32, shape=[32])

    sar_model = SARModel(95)
    encoder_state, feature_map, mask_map = sar_model.inference(input_images, input_widths, batch_size=32)
    logits, att_weights, pred = sar_model.decode(encoder_state, feature_map, input_labels, mask_map)
    loss = sar_model.loss(logits, input_labels, input_lengths)

    optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
    import numpy as np
    _input_images = np.random.rand(32, 48, 160, 3)
    _input_labels = np.random.randint(0,95,size=[32,40])
    _input_lenghts = np.random.randint(0,2,size=[32,40])
    _input_feature_map = np.random.rand(32, 12, 20, 512)
    _input_images_width = np.random.randint(10, 30, 32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            _, loss_value = sess.run([optimizer, loss], feed_dict={input_images: _input_images,
                                                                   input_labels: _input_labels,
                                                                   input_lengths: _input_lenghts,
                                                                   input_feature_map: _input_feature_map,
                                                                   input_widths: _input_images_width})
            print(loss_value)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    test()