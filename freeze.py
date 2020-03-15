import sys
import os
import tensorflow as tf
import numpy as np

from sar_model import SARModel
from data_provider.data_utils import get_vocabulary
from config import get_args

def main_train(args):

    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    # Build graph
    input_train_images = tf.placeholder(dtype=tf.float32, shape=[None, args.height, args.width, 3], name="input_train_images")
    input_train_images_width = tf.placeholder(dtype=tf.float32, shape=[None], name="input_train_width")
    input_train_labels = tf.placeholder(dtype=tf.int32, shape=[None, args.max_len], name="input_train_labels")
    input_train_labels_mask = tf.placeholder(dtype=tf.int32, shape=[None, args.max_len], name="input_train_labels_mask")

    input_val_images = tf.placeholder(dtype=tf.float32, shape=[None, args.height, args.width, 3],name="input_val_images")
    input_val_images_width = tf.placeholder(dtype=tf.float32, shape=[None], name="input_val_width")
    input_val_labels = tf.placeholder(dtype=tf.int32, shape=[None, args.max_len], name="input_val_labels")
    input_val_labels_mask = tf.placeholder(dtype=tf.int32, shape=[None, args.max_len], name="input_val_labels_mask")

    sar_model = SARModel(num_classes=len(voc),
                        encoder_dim=args.encoder_sdim,
                        encoder_layer=args.encoder_layers,
                        decoder_dim=args.decoder_sdim,
                        decoder_layer=args.decoder_layers,
                        decoder_embed_dim=args.decoder_edim,
                        seq_len=args.max_len,
                        is_training=True)
    sar_model_val = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)
    train_model_infer, train_attention_weights, train_pred = sar_model(input_train_images, input_train_labels,
                                                                       input_train_images_width,
                                                                       reuse=False)
    train_loss = sar_model.loss(train_model_infer, input_train_labels, input_train_labels_mask)

    val_model_infer, val_attention_weights, val_pred = sar_model_val(input_val_images, input_val_labels,
                                                                       input_val_images_width,
                                                                       reuse=True)
    val_loss = sar_model_val.loss(val_model_infer, input_val_labels, input_val_labels_mask)

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=args.lr,
                                               global_step=global_step,
                                               decay_steps=args.decay_iter,
                                               decay_rate=args.weight_decay,
                                               staircase=True)

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = optimizer.compute_gradients(train_loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)



    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    variables_to_restore = variable_averages.variables_to_restore()


    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    saver = tf.train.Saver(variables_to_restore)
    summary_writer = tf.summary.FileWriter(args.checkpoints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer.add_graph(sess.graph)
        start_iter = 0
        if args.checkpoints != '':
            print('Restore model from {:s}'.format(args.checkpoints))
            ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
            model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess=sess, save_path=model_path)
            start_iter = sess.run(tf.train.get_global_step())

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                ['sar_1/ArgMax']  # The output node names are used to select the useful nodes
            )

            frozen_model_path = os.path.join(args.checkpoints,os.path.basename(ckpt_state.model_checkpoint_path))+".pb"

            with tf.gfile.GFile(frozen_model_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())
        print("Frozen model saved at " + frozen_model_path)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_train(args)
