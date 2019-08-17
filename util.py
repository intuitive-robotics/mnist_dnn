import tensorflow as tf
import numpy as np
from PIL import Image


def get_label_from_path(DATA_PATH_LIST):
	LABEL_LIST = []
	for path in DATA_PATH_LIST:
		label = path.split('/')[-2]
		LABEL_LIST.append(label)
	return LABEL_LIST

def Data_Path2Data(DATA_PATH_LIST, LABEL_LIST):

	image = np.array(Image.open(DATA_PATH_LIST))
	image.reshape(image.shape[0], image.shape[1], 1)
	
	LABEL_LIST = np.array(LABEL_LIST, dtype=np.uint8)
	return image.astype(np.int32), LABEL_LIST

def data_slice_and_batch(DATA_PATH_LIST, LABEL_LIST, shuffle_buffer_size, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((DATA_PATH_LIST, LABEL_LIST))
	dataset = dataset.map(lambda DATA_PATH_LIST, LABEL_LIST:
		tuple(tf.py_func(Data_Path2Data, [DATA_PATH_LIST, LABEL_LIST],
			[tf.int32, tf.uint8])))

	dataset = dataset.shuffle(buffer_size=(shuffle_buffer_size))
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)

	return dataset

