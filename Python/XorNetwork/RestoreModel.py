import tensorflow as tf
import numpy as np

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_Y = np.array([[0], [1], [1], [0]])

sess=tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'x_input'
output_key = 'y_output'

export_path =  './model'
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)

y_ = tf.placeholder(tf.float32, shape=[None, 1])


print(sess.run(y, feed_dict={x: XOR_X[0].reshape(1,2)}))
print(sess.run(y, feed_dict={x: XOR_X[1].reshape(1,2)}))
print(sess.run(y, feed_dict={x: XOR_X[2].reshape(1,2)}))
print(sess.run(y, feed_dict={x: XOR_X[3].reshape(1,2)}))