# from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/03_polynomial_regression.py
# Simple tutorial for using TensorFlow to compute polynomial regression.
# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# Let's create some toy data
plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()

# tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float32, name="x")
Y = tf.placeholder(tf.float32)


# Instead of a single factor and a bias, we'll create a polynomial function
# of different polynomial degrees.  We will then learn the influence that each
# degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).

powRange = 5
W = tf.Variable(tf.random_normal([powRange-1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


def funct(i):
    return tf.multiply(tf.pow(X, i), W[i-1])

powers = np.array([funct(i) for i in range(1, powRange)]).sum()
Y_pred = tf.add(powers, b, name='Y_pred')

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# %% if we wanted to add regularization, we could add other terms to the cost,
# e.g. ridge regression has a parameter controlling the amount of shrinkage
# over the norm of activations. the larger the shrinkage, the more robust
# to collinearity.
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Saver op to save and restore all the variables
saver = tf.train.Saver()

# %% We create a session to use the graph
n_epochs = 750
with tf.Session() as sess:

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    # once this is created execute
    #   python -m tensorflow.tensorboard --logdir /Users/boris/Projects/TensorFlowPython/TensorFlowTutorials//logs/
    # go to localhost:6006 to watch tensorboard
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 1.0
    merged = tf.summary.merge_all()

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        if epoch_i % 100 == 0:
            print(epoch_i, training_cost)

        if epoch_i % 100 == 0:
            current = Y_pred.eval(feed_dict={X: xs}, session=sess)
            ax.plot(xs, current,
                    'k', alpha=epoch_i / n_epochs)
        fig.show()
        plt.draw()

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
    print "bias ", sess.run(b),  "weight ", sess.run(W)

    # Save produced model
    model_path = "/Users/boris/Projects/TensorFlowPython/models/"
    model_name = "PolynomialRegression"
    save_path = saver.save(sess, model_path+model_name+".ckpt")
    print "Saved model at ", save_path
    graph_path = tf.train.write_graph(sess.graph_def, model_path, model_name+".pb", as_text=True)
    print "Saved graph at :", graph_path

# Now freeze the graph (put variables into graph)

input_saver_def_path = ""
input_binary = False
output_node_names = "Y_pred"            # Model result node
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = model_path + 'frozen_' + model_name + '.pb'
clear_devices = True


freeze_graph.freeze_graph(graph_path, input_saver_def_path,
                          input_binary, save_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
print "Model is frosen"

# optimizing graph

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ["x"], # an array of the input node(s)
    ["Y_pred"], # an array of output nodes
    tf.float32.as_datatype_enum)

# Save the optimized graph

tf.train.write_graph(input_graph_def, model_path, "optimized_" + model_name + ".pb", as_text=False)
tf.train.write_graph(input_graph_def, model_path, "optimized_text_" + model_name + ".pb", as_text=True)


ax.set_ylim([-3, 3])
fig.show()
plt.waitforbuttonpress()