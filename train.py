import mxnet as mx
import logging
import argparse
import os

parser = argparse.ArgumentParser(description='train an face classifer using DeepId model')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
# I only have one gpu(GTX970)
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')


# I still don't know what kv_store's function here
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')



# About learnning rate 
parser.add_argument('--lr', type=float, default=.01,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
# How to use?
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')

parser.add_argument('--num-examples', type=int, default=int(2140 * 0.8),
                    help='the number of training examples')
# need log to file?
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')

parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
# todo statistic about mean data
args = parser.parse_args()

import symbol
net = symbol.get_symbol(output_dim = 30)

from data import FileIter

train = FileIter(
         eval_ratio = 0.2, 
         is_val = False,
         data_name = "data",
         batch_size = args.batch_size,
         label_name = "lr_label"
        )

val = FileIter(
     eval_ratio = 0.2, 
     is_val = True,
     data_name = "data",
     batch_size = 1,
     label_name = "lr_label"
    )


from solver import Solver
model = Solver(
    symbol = net,
    num_epoch = args.num_epochs
    )
model.fit(train, val, 
    batch_end_callback = mx.callback.Speedometer(1, 10),
    epoch_end_callback = mx.callback.do_checkpoint(args.model_prefix))
