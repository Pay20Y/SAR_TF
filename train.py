import sys
import os
import time
import tensorflow as tf
import numpy as np

from sar_model import SARModel
# from data_provider.data_generator import get_batch
# from data_provider.lmdb_data_generator import get_batch
from data_provider import data_generator
from data_provider import lmdb_data_generator
from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from config import get_args

def get_data(image_dir, gt_path, voc_type, max_len, num_samples, height, width, batch_size, workers, keep_ratio, with_aug):
    data_list = []
    if isinstance(image_dir, list) and len(image_dir) > 1:
        # assert len(image_dir) == len(gt_path), "datasets and gt are not corresponding"
        assert batch_size % len(image_dir) == 0, "batch size should divide dataset num"
        per_batch_size = batch_size // len(image_dir)
        if None in gt_path:
            # Using lmdb input
            for i in image_dir:
                data_list.append(lmdb_data_generator.get_batch(workers, lmdb_dir=i, input_height=height, input_width=width, batch_size=per_batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug))
        else:
            for i, g in zip(image_dir, gt_path):
                data_list.append(data_generator.get_batch(workers, image_dir=i, gt_path=g, input_height=height, input_width=width, batch_size=per_batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug))
    else:
        if isinstance(image_dir, list):
            if None in gt_path:
                data = lmdb_data_generator.get_batch(workers, lmdb_dir=image_dir[0], input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
            else:
                data = data_generator.get_batch(workers, image_dir=image_dir[0], gt_path=gt_path[0], input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
        else:
            if gt_path is None:
                data = lmdb_data_generator.get_batch(workers, lmdb_dir=image_dir, input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
            else:
                data = data_generator.get_batch(workers, image_dir=image_dir, gt_path=gt_path, input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
        data_list.append(data)

    return data_list

def get_batch_data(data_list, batch_size):
    batch_images = []
    batch_labels = []
    batch_labels_mask = []
    batch_labels_str = []
    batch_widths = []
    for data in data_list:
        _data = next(data)
        batch_images.append(_data[0])
        batch_labels.append(_data[1])
        batch_labels_mask.append(_data[2])
        batch_labels_str.extend(_data[4])
        batch_widths.append(_data[5])

    batch_images = np.concatenate(batch_images, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)
    batch_labels_mask = np.concatenate(batch_labels_mask, axis=0)
    batch_widths = np.concatenate(batch_widths, axis=0)

    assert len(batch_images) == batch_size, "concat data is not equal to batch size"

    return batch_images, batch_labels, batch_labels_mask, batch_labels_str, batch_widths
def get_batch_data_dummy(batch_size, height, width, max_len):
    batch_images = np.random.rand(batch_size, height, width, 3)
    batch_labels = np.random.randint(0, 97, [batch_size, max_len])
    batch_masks = np.random.randint(0, 2, [batch_size, max_len])
    return batch_images, batch_labels, batch_masks
def main_train(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    # Build graph
    input_train_images = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.height, args.width, 3], name="input_train_images")
    input_train_images_width = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size], name="input_train_width")
    input_train_labels = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels")
    input_train_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels_mask")

    input_val_images = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size, args.height, args.width, 3],name="input_val_images")
    input_val_images_width = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size], name="input_val_width")
    input_val_labels = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels")
    input_val_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels_mask")

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
                                                                       batch_size=args.train_batch_size, reuse=False)
    train_loss = sar_model.loss(train_model_infer, input_train_labels, input_train_labels_mask)

    val_model_infer, val_attention_weights, val_pred = sar_model_val(input_val_images, input_val_labels,
                                                                       input_val_images_width,
                                                                       batch_size=args.val_batch_size, reuse=True)
    val_loss = sar_model_val.loss(val_model_infer, input_val_labels, input_val_labels_mask)

    train_data_list = get_data(args.train_data_dir,
                         args.train_data_gt,
                         args.voc_type,
                         args.max_len,
                         args.num_train,
                         args.height,
                         args.width,
                         args.train_batch_size,
                         args.workers,
                         args.keep_ratio,
                         with_aug=args.aug)

    val_data_list = get_data(args.test_data_dir,
                         args.test_data_gt,
                         args.voc_type,
                         args.max_len,
                         args.num_train,
                         args.height,
                         args.width,
                         args.val_batch_size,
                         args.workers,
                         args.keep_ratio,
                         with_aug=False)

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

    # Save summary
    os.makedirs(args.checkpoints, exist_ok=True)
    tf.summary.scalar(name='train_loss', tensor=train_loss)
    tf.summary.scalar(name='val_loss', tensor=val_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    merge_summary_op = tf.summary.merge_all()

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'sar_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(args.checkpoints, model_name)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    summary_writer = tf.summary.FileWriter(args.checkpoints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer.add_graph(sess.graph)
        start_iter = 0
        if args.resume == True and args.pretrained != '':
            print('Restore model from {:s}'.format(args.pretrained))
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess=sess, save_path=model_path)
            start_iter = sess.run(tf.train.get_global_step())
        else:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)

        while start_iter < args.iters:
            start_iter += 1
            train_data = get_batch_data(train_data_list, args.train_batch_size)
            _, train_loss_value, train_pred_value = sess.run([train_op, train_loss, train_pred], feed_dict={input_train_images: train_data[0],
                                                                               input_train_labels: train_data[1],
                                                                               input_train_labels_mask: train_data[2],
                                                                                input_train_images_width: train_data[4]})

            if start_iter % args.log_iter == 0:
                print("Iter {} train loss= {:3f}".format(start_iter, train_loss_value))

            if start_iter % args.summary_iter == 0:
                val_data = get_batch_data(val_data_list, args.val_batch_size)

                merge_summary_value, val_pred_value, val_loss_value  = sess.run([merge_summary_op, val_pred, val_loss], feed_dict={input_train_images: train_data[0],
                                                                                                         input_train_labels: train_data[1],
                                                                                                         input_train_labels_mask: train_data[2],
                                                                                                        input_train_images_width: train_data[4],
                                                                                                         input_val_images: val_data[0],
                                                                                                         input_val_labels: val_data[1],
                                                                                                         input_val_labels_mask: val_data[2],
                                                                                                        input_val_images_width: val_data[4]})

                summary_writer.add_summary(summary=merge_summary_value, global_step=start_iter)
                if start_iter % args.eval_iter == 0:
                    print("#" * 80)
                    print("train prediction \t train labels ")
                    for result, gt in zip(idx2label(train_pred_value), train_data[3]):
                        print("{} \t {}".format(result, gt))
                    print("#" * 80)
                    print("test prediction \t test labels ")
                    for result, gt in zip(idx2label(val_pred_value), val_data[3]):
                        print("{} \t {}".format(result, gt))
                    print("#" * 80)

                    train_metrics_result = calc_metrics(idx2label(train_pred_value), train_data[3], metrics_type="accuracy")
                    val_metrics_result = calc_metrics(idx2label(val_pred_value), val_data[3], metrics_type="accuracy")
                    print("Evaluation Iter {} test loss: {:3f} train accuracy: {:3f} test accuracy {:3f}".format(start_iter,
                                                                                                                 val_loss_value,
                                                                                                                 train_metrics_result,
                                                                                                                 val_metrics_result))
            if start_iter % args.save_iter == 0:
                print("Iter {} save to checkpoint".format(start_iter))
                saver.save(sess, model_save_path, global_step=global_step)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_train(args)