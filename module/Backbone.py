import tensorflow as tf
from tensorflow.contrib import slim
# from module.nets import resnet_utils, resnet_v1

def mean_image_subtraction(images, means=[128.0, 128.0, 128.0]):
	'''
	image normalization
	:param images:
	:param means:
	:return:
	'''
	with tf.variable_scope("mean_subtraction"):
		num_channels = images.get_shape().as_list()[-1]
		if len(means) != num_channels:
			raise ValueError('len(means) must match the number of channels')
		channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
		for i in range(num_channels):
			channels[i] -= means[i]
			return tf.concat(axis=3, values=channels)

class Backbone(object):
	def __init__(self, weight_decay=1e-5, is_training=True):
		self.weight_decay = weight_decay
		self.is_training = is_training

	def shortcut(self, inputs, output_dim, stride=1):
		depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
		if depth_in == output_dim:
			shortcut = inputs
		else:
			shortcut = slim.conv2d(inputs, output_dim, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
		return shortcut

	def basicblock(self, inputs, output_dim, kernel_size, stride, scope):
		with tf.variable_scope(scope):
			shortcut = self.shortcut(inputs, output_dim)
			residual = slim.conv2d(inputs, output_dim, kernel_size, stride=stride, scope='conv1')
			residual = slim.conv2d(residual, output_dim, kernel_size, stride=stride, activation_fn=None, scope='conv2')

			return tf.nn.relu(shortcut + residual)

	def __call__(self, input_image):
		with tf.variable_scope("resnet_backbone"):
			batch_norm_params = {
				'decay': 0.997,
				'epsilon': 1e-5,
				'scale': True,
				'is_training': self.is_training
			}
			with slim.arg_scope([slim.conv2d],
								activation_fn=tf.nn.relu,
								normalizer_fn=slim.batch_norm,
								normalizer_params=batch_norm_params,
								weights_regularizer=slim.l2_regularizer(self.weight_decay)):
				input_image = mean_image_subtraction(input_image)
				net = slim.conv2d(input_image, 64, 3, stride=1, rate=1, padding='SAME', scope="conv1")
				net = slim.conv2d(net, 128, 3, stride=1, rate=1, padding='SAME', scope="conv2")
				net = slim.max_pool2d(net, [2, 2], stride=2)
				# 1 Block here
				net = self.basicblock(net, output_dim=256, kernel_size=3, stride=1, scope="block1")
				# 1 Block end
				net = slim.conv2d(net, 256, 3, stride=1, rate=1, padding='SAME', scope="conv3")
				net = slim.max_pool2d(net, [2, 2], stride=2)
				# 2 Blocks here
				net = self.basicblock(net, output_dim=256, kernel_size=3, stride=1, scope="block2")
				net = self.basicblock(net, output_dim=256, kernel_size=3, stride=1, scope="block3")
				# 2 Blocks end
				net = slim.conv2d(net, 256, 3, stride=1, rate=1, padding='SAME', scope="conv4")
				net = slim.max_pool2d(net, [2, 1], stride=[2, 1])
				# 5 Blocks here
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block4")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block5")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block6")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block7")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block8")
				# 5 Blocks end
				net = slim.conv2d(net, 512, 3, stride=1, rate=1, padding='SAME', scope="conv5")
				# 3 Blocks here
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block9")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block10")
				net = self.basicblock(net, output_dim=512, kernel_size=3, stride=1, scope="block11")
				# 3 Blocks end
				net = slim.conv2d(net, 512, 3, stride=1, rate=1, padding='SAME', scope="conv6")

				print("Backbone final output: ", net)

				return net


if __name__ == '__main__':
	import numpy as np
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = "1"
	bb = Backbone()
	input_images = tf.placeholder(tf.float32, shape=[32, 48 ,160, 3], name="input_images")
	input_feature_label = tf.placeholder(tf.float32, shape=[32, 6, 40, 512], name="input_feature_label")
	feature_map = bb(input_images)
	loss = tf.reduce_mean(input_feature_label - feature_map)
	print("feature_map: ", feature_map)
	optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)

	_input_images = np.random.rand(32, 48, 160, 3)
	_input_feature_label = np.random.rand(32, 6, 40, 512)
	summary_writer = tf.summary.FileWriter("toy_summary")
	with tf.Session()  as sess:
		summary_writer.add_graph(sess.graph)
		sess.run(tf.global_variables_initializer())
		for i in range(10):
			_, loss_value = sess.run([optimizer, loss], feed_dict={input_images: _input_images, input_feature_label: _input_feature_label})
			print(loss_value)