import tensorflow as tf
import numpy as np
import cv2
import os
import sys

from data_provider.data_utils import get_vocabulary
from data_provider.json_loader import JSONLoader
from data_provider.text_loader import TextLoader
from utils.transcription_utils import idx2label, calc_metrics
from sar_model import SARModel
from utils.metrics import accuracy

from config import get_args

def get_images(images_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(images_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def get_data(args):
    if args.test_data_gt != '' and args.test_data_gt is not None:
        if args.test_data_gt.split('.')[1] == 'json':
            # data_loader = ArTLoader(args.test_data_dir)
            data_loader = TextLoader(args.test_data_dir)
        elif args.test_data_gt.split('.')[1] == 'txt':
            data_loader = JSONLoader(args.test_data_dir)
        else:
            raise Exception("Unsupported file type")
        images_path, labels = data_loader.parse_gt(args.test_data_gt)
        return images_path, labels
    else:
        images_path = get_images(args.test_data_dir)
        labels = ['' for i in range(len(images_path))]
        return images_path, labels

def data_preprocess(image, word, char2id, args):
    H, W, C = image.shape
    # Rotate the vertical images
    if H > 4 * W:
        image = np.rot90(image)
        H, W = W, H

    new_width = int((1.0 * args.height / H) * W)
    new_width = new_width if new_width < args.width else args.width
    new_height = args.height
    img_resize = np.zeros((args.height, args.width, C), dtype=np.uint8)
    image = cv2.resize(image, (new_width, new_height))
    img_resize[:, :new_width, :] = image

    label = np.full((args.max_len), char2id['PAD'], dtype=np.int)
    label_list = []
    for char in word:
        if char in char2id:
            label_list.append(char2id[char])
        else:
            label_list.append(char2id['UNK'])
    # label_list = label_list + [char2id['EOS']]
    # assert len(label_list) <= max_len
    if len(label_list) > (args.max_len - 1):
        label_list = label_list[:(args.max_len - 1)]
    label_list = label_list + [char2id['EOS']]
    label[:len(label_list)] = np.array(label_list)

    return img_resize, label, new_width

def main_test(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    input_images = tf.placeholder(dtype=tf.float32, shape=[1, args.height, None, 3], name="input_images")
    input_images_width = tf.placeholder(dtype=tf.float32, shape=[1], name="input_images_width")
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, args.max_len], name="input_labels")
    sar_model = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)

    model_infer, attention_weights, pred = sar_model(input_images, input_labels, input_images_width, batch_size=1, reuse=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
        model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

        images_path, labels = get_data(args)
        predicts = []
        for img_path, label in zip(images_path, labels):
            try:
                img = cv2.imread(img_path)
            except Exception as e:
                print("{} error: {}".format(img_path, e))
                continue

            img, la, width = data_preprocess(img, label, char2id, args)

            pred_value, attention_weights_value = sess.run([pred, attention_weights], feed_dict={input_images: [img],
                                                                                                 input_labels: [la],
                                                                                                 input_images_width: [width]})
            pred_value_str = idx2label(pred_value, id2char, char2id)[0]
            print("predict: {} label: {}".format(pred_value_str, label))
            predicts.append(pred_value_str)
            pred_value_str += '$'
            if args.vis_dir != None and args.vis_dir != "":
                os.makedirs(args.vis_dir, exist_ok=True)
                assert len(img.shape) == 3
                att_maps = attention_weights_value.reshape([-1, attention_weights_value.shape[2], attention_weights_value.shape[3], 1]) # T * H * W * 1
                for i, att_map in enumerate(att_maps):
                    if i >= len(pred_value_str):
                        break
                    att_map = cv2.resize(att_map, (img.shape[1], img.shape[0]))
                    _att_map = np.zeros(dtype=np.uint8, shape=[img.shape[0], img.shape[1], 3])
                    _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

                    show_attention = cv2.addWeighted(img, 0.5, _att_map, 2, 0)
                    cv2.imwrite(os.path.join(args.vis_dir, os.path.basename(img_path).split('.')[0] + "_" + str(i) + "_" + pred_value_str[i] + ".jpg"), show_attention)

    acc_rate = accuracy(predicts, labels)
    print("Done, Accuracy: {}".format(acc_rate))

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_test(args)