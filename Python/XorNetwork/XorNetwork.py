import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


MODEL_NAME = "xor_graph"
input_name = "input"
output_name = "output"

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_Y = np.array([[0], [1], [1], [0]])


def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
                              'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, [input_node_names], [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


# Create model

x = tf.placeholder(tf.float32, shape=[None, 2], name=input_name)
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y_")

W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W1")
W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="W2")

B1 = tf.Variable(tf.zeros([2]), name="B1")
B2 = tf.Variable(tf.zeros([1]), name="B2")

z = tf.sigmoid(tf.matmul(x, W1) + B1, name="z")
y = tf.sigmoid(tf.matmul(z, W2) + B2, name=output_name)

cost = tf.reduce_mean(((y_ * tf.log(y)) + ((1 - y_) * tf.log(1.0 - y))) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)
    saver = tf.train.Saver()

    # Train
    for i in range(1000000):
        sess.run(train_step, feed_dict={x: XOR_X, y_: XOR_Y})

        if i % 1000 == 0:
            print('Epoch ', i)
            print('cost ', sess.run(cost, feed_dict={x: XOR_X, y_: XOR_Y}))

    saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

    print(sess.run(y, feed_dict={x: XOR_X[0].reshape(1, 2)}))
    print(sess.run(y, feed_dict={x: XOR_X[1].reshape(1, 2)}))
    print(sess.run(y, feed_dict={x: XOR_X[2].reshape(1, 2)}))
    print(sess.run(y, feed_dict={x: XOR_X[3].reshape(1, 2)}))

export_model(input_name, output_name)

# Save model
export_path = './model'
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'x_input': tensor_info_x},
      outputs={'y_output': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature
  },
  )

builder.save()

