import tensorflow as tf
import numpy as np

from glob import glob
from PIL import Image

from util import get_label_from_path
from util import data_slice_and_batch

# 1.Hyper Parameter
num_epoch = 2
batch_size = 500
height = 28
width = 28
channels = 1
num_classes = 10
shuffle_buffer_size = 100000
shuffle_buffer_size_test = 20000

# 2-1. Import training dataset
DATA_PATH_LIST = glob('./mnist_png/training/*/*.png')
LABEL_LIST = get_label_from_path(DATA_PATH_LIST)
dataset = data_slice_and_batch(DATA_PATH_LIST, LABEL_LIST, shuffle_buffer_size, batch_size)

# 2-2. Import test dataset
TEST_DATA_PATH_LIST = glob('./mnist_png/testing/*/*.png')
TEST_LABEL_LIST = get_label_from_path(TEST_DATA_PATH_LIST)
dataset_test = data_slice_and_batch(TEST_DATA_PATH_LIST, TEST_LABEL_LIST, shuffle_buffer_size_test, batch_size)


# 3.Model Design
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


total_batch = int(len(DATA_PATH_LIST)/batch_size)

for epoch in range(num_epoch):
	total_cost = 0
	iterator = dataset.make_initializable_iterator()
	sess.run(iterator.initializer) # Shuffle이 실행됨

	for i in range(total_batch):
		# batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		image, label = sess.run(iterator.get_next())

		batch_xs = image.reshape(batch_size,784)
		batch_ys = tf.one_hot(label, depth=num_classes).eval(session=sess)
		# print(batch_ys)

		a, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
		total_cost += cost_val

		if i % 10 == 0:
			print('batch:', '%04d' % (i*batch_size), 'cost =', '{:.8f}'.format(cost_val))

	print('Epoch:', '%04d' % (epoch+1), 'Avg. cost =', '{:.3f}'.format(total_cost/total_batch))

print('learning finished!!')

iterator_test = dataset_test.make_initializable_iterator()
sess.run(iterator_test.initializer) # Shuffle이 실행됨
image_test, label_test = sess.run(iterator_test.get_next())

batch_xs_test = image_test.reshape(batch_size,784)
batch_ys_test = tf.one_hot(label_test, depth=num_classes).eval(session=sess)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: ', sess.run(accuracy, feed_dict={X: batch_xs_test, Y: batch_ys_test, keep_prob: 1}))

sess.close()







