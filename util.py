import tensorflow as tf
import numpy as np
from PIL import Image


def get_label_from_path(DATA_PATH_LIST):
	label_list = []
	for path in DATA_PATH_LIST:
		label = path.split('/')[-2]
		label_list.append(label)
	return label_list

def _read_py_function(DATA_PATH_LIST, LABEL_LIST):

	image = np.array(Image.open(DATA_PATH_LIST))
	image.reshape(image.shape[0], image.shape[1], 1)
	
	LABEL_LIST = np.array(LABEL_LIST, dtype=np.uint8)
	return image.astype(np.int32), LABEL_LIST

def data_slice_and_batch(path_list, labels, bufferSize, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((path_list, labels))
	dataset = dataset.map(lambda path_list, labels:
		tuple(tf.py_func(_read_py_function, [path_list, labels],
			[tf.int32, tf.uint8])))

	dataset = dataset.shuffle(buffer_size=(bufferSize))
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)

	return dataset

