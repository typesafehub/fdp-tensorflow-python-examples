# from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/04_logistic_regression.py
# In statistics, logistic regression, or logit regression, or logit model is a regression model where
# the dependent variable (DV) is categorical.
# Logistic regression can be binomial, ordinal or multinomial. Binomial or binary logistic regression deals with
# situations in which the observed outcome for a dependent variable can have only two possible types, "0" and "1"
# (which may represent, for example, "dead" vs. "alive" or "win" vs. "loss"). Multinomial logistic regression deals
# with situations where the outcome can have three or more possible types (e.g., "disease A" vs. "disease B" vs.
# "disease C") that are not ordered. Ordinal logistic regression deals with dependent variables that are ordered.
# In binary logistic regression, the outcome is usually coded as "0" or "1", as this leads to the most straightforward
# interpretation.
# Like other forms of regression analysis, logistic regression makes use of one or more predictor variables that may
# be either continuous or categorical. Unlike ordinary linear regression, however, logistic regression is used for
# predicting binary dependent variables (treating the dependent variable as the outcome of a Bernoulli trial) rather
# than a continuous outcome. Given this difference, the assumptions of linear regression are violated. In particular,
# the residuals cannot be normally distributed. In addition, linear regression may make nonsensical predictions for a
# binary dependent variable. What is needed is a way to convert a binary variable into a continuous one that can take
# on any real value (negative or positive). To do that logistic regression first takes the odds of the event happening
# for different levels of each independent variable, then takes the ratio of those odds (which is continuous but
# cannot be negative) and then takes the logarithm of that ratio. This is referred to as logit or log-odds) to create
# a continuous criterion as a transformed version of the dependent variable.

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# get the classic mnist dataset
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
# https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/download/index.html#dataset-object

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# mnist is now a DataSet with accessors for:
# 'train', 'test', and 'validation'.
# within each, we can access:
# images, labels, and num_examples
print(mnist.train.num_examples,
      mnist.test.num_examples,
      mnist.validation.num_examples)

# %% the images are stored as:
# n_observations x n_features tensor (n-dim array)
# the labels are stored as n_observations x n_labels,
# where each observation is a one-hot vector.
print(mnist.train.images.shape, mnist.train.labels.shape)

# the range of the values of the images is from 0-1
print(np.min(mnist.train.images), np.max(mnist.train.images))

# we can visualize any one of the images by reshaping it to a 28x28 image
plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')
plt.show()
plt.waitforbuttonpress()

# We can create a container for an input image using tensorflow's graph:
# We allow the first dimension to be None, since this will eventually
# represent our mini-batches, or how many images we feed into a network
# at a time during training/validation/testing.
# The second dimension is the number of features that the image has.
n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input], name="net_input")

# We can write a simple regression (y = W*x + b) as:
W = tf.Variable(tf.zeros([n_input, n_output]), name="Weights")
b = tf.Variable(tf.zeros([n_output]), name="bias")
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b, name="net_output")

# We'll create a placeholder for the true output of the network
y_true = tf.placeholder(tf.float32, [None, 10])

# And then write our loss function:
cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))

# This would equate each label in our one-hot vector between the
# prediction and actual using the argmax as the predicted label
correct_prediction = tf.equal(
    tf.argmax(net_output, 1), tf.argmax(y_true, 1))

# %% And now we can look at the mean of our network's correct guesses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# We can tell the tensorflow graph to train w/ gradient descent using
# our loss function and an input learning rate
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Saver op to save and restore all the variables
saver = tf.train.Saver()

# We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Now actually do some training:
batch_size = 100
n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            net_input: batch_xs,
            y_true: batch_ys
        })
    print(sess.run(accuracy,
                   feed_dict={
                       net_input: mnist.validation.images,
                       y_true: mnist.validation.labels
                   }))

# Print final test accuracy:
print(sess.run(accuracy,
               feed_dict={
                   net_input: mnist.test.images,
                   y_true: mnist.test.labels
               }))
print "weight ", sess.run(W), " bias ", sess.run(b)

# Save produced model
model_path = "/Users/boris/Projects/TensorFlowPython/models/"
model_name = "LogisticRegression"
save_path = saver.save(sess, model_path+model_name+".ckpt")
print "Saved model at ", save_path
graph_path = tf.train.write_graph(sess.graph_def, model_path, model_name+".pb", as_text=True)
print "Saved graph at :", graph_path

# Now freeze the graph (put variables into graph)

input_saver_def_path = ""
input_binary = False
output_node_names = "net_output"            # Model result node
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
    ["net_input"], # an array of the input node(s)
    ["net_output"], # an array of output nodes
    tf.float32.as_datatype_enum)

# Save the optimized graph

tf.train.write_graph(output_graph_def, model_path, "optimized_" + model_name + ".pb", as_text=True)

# %%
"""
# We could do the same thing w/ Keras like so:
from keras.models import Sequential
model = Sequential()
from keras.layers.core import Dense, Activation
model.add(Dense(output_dim=10, input_dim=784, init='zero'))
model.add(Activation("softmax"))
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', 
    optimizer=SGD(lr=learning_rate))
model.fit(mnist.train.images, mnist.train.labels, nb_epoch=n_epochs,
          batch_size=batch_size, show_accuracy=True)
objective_score = model.evaluate(mnist.test.images, mnist.test.labels,
                                 batch_size=100, show_accuracy=True)
"""