import tensorflow as tf
from keras.metrics import categorical_accuracy
import numpy as np

y_test = np.ones(2)
y_pred = np.ones(2)

y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

categorical_accuracy(y_test, y_pred)