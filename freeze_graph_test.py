# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py
# Not very usefull

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.tools import freeze_graph

checkpoint_prefix = "models/saved_checkpoint"
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

# We'll create an input graph that has a single variable containing 1.0,
# and that then multiplies it by 2.
with ops.Graph().as_default():
    variable_node = tf.Variable(1.0, name="variable_node")
    output_node = tf.multiply(variable_node, 2.0, name="output_node")
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    output = sess.run(output_node)
#    tf.assertNear(2.0, output, 0.00001)
    saver = tf.train.Saver()
    checkpoint_path = saver.save(
        sess,
        checkpoint_prefix,
        global_step=0,
        latest_filename=checkpoint_state_name)
    input_graph_path = tf.train.write_graph(sess.graph, "models", input_graph_name)

# We save out the graph to disk, and then call the const conversion
# routine.
input_saver_def_path = ""
input_binary = False
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_node_names = "output_node"
output_graph_path = "models/output_graph.pb"
clear_devices = False
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_graph_path, clear_devices, "")