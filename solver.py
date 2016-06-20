import mxnet as mx
import logging
import os
from collections import namedtuple
from mxnet import metric
import pdb

BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])
class Solver(object):
    def __init__(self, symbol, ctx=None,
                 grad_req = "write", initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                 begin_epoch = 0, num_epoch = None,
                 optimizer = 'sgd', **kwargs):
        self.symbol = symbol
        # pdb.set_trace()
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = {}
        self.aux_params = {}
        self.optimizer =  optimizer
        self.initializer = initializer
        self.kwargs = kwargs.copy()

    def fit(self, train_data, eval_data,
        eval_metric='mse',
        grad_req='write',
        epoch_end_callback=None,
        batch_end_callback=None,
        kv_store='local',
        logger=None):

        if logger is None:
            logger = logging
        logging.info('Starting training with %s', str(self.ctx))
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data = train_data.provide_data[0][1])
        arg_names = self.symbol.list_arguments()

        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('label')):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        #init the params
        pdb.set_trace()
        self.arg_params = {k : mx.nd.zeros(s, self.ctx) for k, s in zip(arg_names, arg_shapes)}
        for k, v in self.arg_params.items():
            if not (k.endswith('data') or k.endswith('label')):
                self.initializer(k, v)

        #init the aux params
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k : mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

        data_name = train_data.data_name
        label_name = train_data.label_name
        input_names = [data_name, label_name]

        self.optimizer = mx.optimizer.create(self.optimizer, rescale_grad = (1.0/train_data.get_batch_size()), **(self.kwargs))
        self.updater = mx.optimizer.get_updater(self.optimizer)

        eval_metric = metric.create(eval_metric)

        # begin training
        for epoch in range(self.begin_epoch, self.num_epoch):
            nbatch = 0
            train_data.reset()
            eval_metric.reset()

            #train
            for databatch in train_data:
                nbatch += 1
                for k, v in databatch.data.items():
                    self.arg_params[k] = mx.nd.array(v, self.ctx)
                for k, v in databatch.label.items():
                    self.arg_params[k] = mx.nd.array(v, self.ctx)
                executor = self.symbol.bind(self.ctx, self.arg_params, args_grad = self.grad_params,
                    grad_req = grad_req, aux_states = self.aux_params)
                # print(nbatch)
                if nbatch == 1550:
                    pdb.set_trace()
                update_dict = {name:nd for name, nd
                                in zip(self.symbol.list_arguments(), executor.grad_arrays) if nd}
                output_dict = {name:nd for name, nd
                                in zip(self.symbol.list_outputs(), executor.outputs)}
                # pdb.set_trace()
                executor.forward(is_train=True)
                executor.backward()

                for key, arr in update_dict.items():
                    self.updater(key, arr, self.arg_params[key])

                label = self.arg_params['lr_label']
                pred = output_dict['lr_output']
                eval_metric.update([label], [pred])
                executor.outputs[0].wait_to_read()
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric)
                batch_end_callback(batch_end_params)
            if epoch_end_callback != None:
                epoch_end_callback(epoch, self.symbol, self.arg_params, self.aux_params)
            # pdb.set_trace()
            name, value = eval_metric.get()
            logger.info("------------------------------>Epoch[%d] Train-%s=%f", epoch, name, value)

            #begin evaluation
            if eval_data:
                logger.info( "in eval process...")
                nbatch = 0
                eval_data.reset()
                eval_metric.reset()
                for data in eval_data:
                    nbatch += 1
                    for k, v in databatch.data.items():
                        self.arg_params[k] = mx.nd.array(v, self.ctx)
                    for k, v in databatch.label.items():
                        self.arg_params[k] = mx.nd.array(v, self.ctx)
                    executor = self.symbol.bind(self.ctx, self.arg_params, args_grad = self.grad_params,
                        grad_req = grad_req, aux_states = self.aux_params)

                    output_dict = {name:nd for name, nd
                                    in zip(self.symbol.list_outputs(), executor.outputs)}
                    executor.forward(is_train=False)
                    label = self.arg_params['lr_label']
                    pred = output_dict['lr_output']
                    eval_metric.update([label], [pred])
            name, value = eval_metric.get()
            logger.info('batch[%d] Validation-%s=%f', nbatch, name, value)











