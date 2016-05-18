import mxnet as mx

def get_symbol(output_dim = 30):
	input_data = mx.symbol.Variable(name='data')

	# group 1
	conv1 = mx.symbol.Convolution(
		data=input_data, kernel=(3, 3), stride=(1, 1), num_filter=32, name="conv1")
	relu1 = mx.symbol.Activation(data=bn1, act_type="relu", name="relu1")
	pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")

	# group 2
	conv2 = mx.symbol.Convolution(
		data=pool1, kernel=(2, 2), stride=(1, 1), num_filter=64, name="conv2")
	relu2 = mx.symbol.Activation(data=bn2, act_type="relu", name="relu2")
	pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")

	# group3
	conv3 = mx.symbol.Convolution(
		data=pool2, kernel=(2, 2), stride=(1, 1), num_filter=128, name="conv3")
	relu3 = mx.symbol.Activation(data=bn3, act_type="relu", name="relu3")
	# pool3 = mx.symbol.Pooling(data=relu3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")

	

	# drop3
	drop3 = mx.symbol.Flatten(data=relu3, name="drop3", p = 0.5)

	# fc (fully connect)
	fc1 = mx.symbol.FullyConnected(data=drop3, num_hidden=500, name="fc1")
	fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=500, name="fc2")
	fc3 = mx.symbol.FullyConnected(data=fc2, num_hidden=output_dim, name="fc3")
	return fc3
