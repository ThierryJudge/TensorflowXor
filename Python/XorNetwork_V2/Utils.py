import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from matplotlib import pyplot as plt


def export_model(input_node_names, output_node_name, model_name):
    freeze_graph.freeze_graph('out/' + model_name + '.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, [input_node_names], [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def generate_data(n):
    data = []
    labels = []
    for i in range(n):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        if (x < 0 and y < 0) or (x >= 0 and y >= 0):    # XOR
            label = [0]
        else:
            label = [1]

        if abs(x) < 0 or abs(y) < 0:
            i = i - 1
        else:
            p = [x, y]
            data.append(p)
            labels.append(label)
    return np.array(data), np.array(labels)


def scatter_plot(data, labels):
    for i in range(len(data)):
        if i % (len(data)/1000) == 0:
            p = data[i]
            c = 'r'
            if labels[i] == 1:
                c = 'b'
            plt.scatter(p[0], p[1], color=c)
    plt.title('Generated Data')
    plt.show()


def get_batch(data, labels, batch_size):
    r = np.random.randint(len(data) - batch_size - 1)
    x = data[r:r + batch_size]
    y = labels[r:r + batch_size]

    return x, y
