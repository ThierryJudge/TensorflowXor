import tensorflow as tf
import numpy as np
from Utils import export_model
from Utils import scatter_plot
from Utils import generate_data
from Utils import get_batch
from matplotlib import pyplot as plt


MODEL_NAME = "xor_graph"
input_name = "input"
output_name = "output"

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_Y = np.array([[0], [1], [1], [0]])


data, labels = generate_data(100000)
test_data, test_labels = generate_data(1000)

scatter_plot(data, labels)


# Create model
x = tf.placeholder(tf.float32, shape=[None, 2], name=input_name)
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y_")

W1 = tf.Variable(tf.random_uniform([2, 10], -1, 1), name="W1")
W2 = tf.Variable(tf.random_uniform([10, 1], -1, 1), name="W2")

B1 = tf.Variable(tf.zeros([10]), name="B1")
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
    for i in range(500000):
        XOR_X, XOR_Y = get_batch(data, labels, 10)
        sess.run(train_step, feed_dict={x: XOR_X, y_: XOR_Y})

        if i % 1000 == 0:
            print('Epoch ', i)
            print('cost ', sess.run(cost, feed_dict={x: XOR_X, y_: XOR_Y}))

    saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

    # Test
    total_tests = len(test_data)
    successes = 0
    failures = 0
    for i in range(len(test_data)):
        target = test_labels[i]
        data_x = test_data[i]

        prediction = sess.run(y, feed_dict={x: data_x.reshape(1, 2)})

        c = "b"
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
            c = 'r'

        plt.scatter(data_x[0], data_x[1], color=c)
        if prediction == target:
            successes = successes + 1
        else:
            failures = failures + 1

    print("Accuracy: " + str(successes / total_tests * 100) + "% for " + str(total_tests) + " tests.")
    plt.title("Tested Data")
    plt.show()


export_model(input_name, output_name, MODEL_NAME)
