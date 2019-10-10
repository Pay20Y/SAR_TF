from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description="Softmax loss classification")

# Data
parser.add_argument('--train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=[None])
parser.add_argument('--train_data_gt', nargs='+', type=str, metavar='PATH',
                    default=[None])
parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                    default=None)
parser.add_argument('--test_data_gt', type=str, metavar='PATH',
                    default=None)
parser.add_argument('-b', '--train_batch_size', type=int, default=128)
parser.add_argument('-v', '--val_batch_size', type=int, default=128)
parser.add_argument('-j', '--workers', type=int, default=2)
parser.add_argument('-g', '--gpus', type=str, default='1')
parser.add_argument('--height', type=int, default=48, help="input height")
parser.add_argument('--width', type=int, default=160, help="input width")
parser.add_argument('--aug', type=bool, default=False, help="using data augmentation or not")
parser.add_argument('--keep_ratio', action='store_true', default=True,
                    help='length fixed or lenghth variable.')
parser.add_argument('--voc_type', type=str, default='ALLCASES_SYMBOLS',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
parser.add_argument('--num_train', type=int, default=-1)
parser.add_argument('--num_test', type=int, default=-1)

# Model
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--encoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in encoder.")
parser.add_argument('--encoder_layers', type=int, default=2,
                    help="the num of layers in encoder lstm.")
parser.add_argument('--decoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in decoder.")
parser.add_argument('--decoder_layers', type=int, default=2,
                    help="the num of layers in decoder lstm.")
parser.add_argument('--decoder_edim', type=int, default=512,
                    help="the dim of embedding layer in decoder.")

# Optimizer
parser.add_argument('--lr', type=float, default=0.0008, # 0.001
                    help="learning rate of new parameters, for pretrained ")
parser.add_argument('--weight_decay', type=float, default=0.9) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--decay_iter', type=int, default=100000)
parser.add_argument('--decay_end', type=float, default=0.00001)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--iters', type=int, default=3000000)
parser.add_argument('--decode_type', type=str, default='greed')

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--pretrained', type=str, default='', metavar='PATH')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', metavar='PATH')
parser.add_argument('--log_iter', type=int, default=100)
parser.add_argument('--summary_iter', type=int, default=1000)
parser.add_argument('--eval_iter', type=int, default=2000)
parser.add_argument('--save_iter', type=int, default=2000)
parser.add_argument('--vis_dir', type=str, metavar='PATH')

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args