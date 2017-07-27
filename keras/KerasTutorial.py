# tutorial from http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Model saving is at https://gist.github.com/ismaeIfm/eeb24fad2623dfb69ca81bb0f254543f
# A good TF/Keras interoperability post https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K


# fix random seed for reproducibility
numpy.random.seed(7)

# create TF session and set it in Keras
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)

# load pima indians dataset
dataset = numpy.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

print('Done training!')

modelInput = model.input
modelOutput = model.output
print "input", modelInput.name
print "output", modelOutput.name

print("Model's parameters")
print(model.get_weights())


K.set_learning_phase(0)  # all new operations will be in test mode from now on

# serialize the model and get its weights, for quick re-building
config = model.to_json()
weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = model_from_json(config)
new_model.set_weights(weights)

#create the saver
# Saver op to save and restore all the variables
saver = tf.train.Saver()

# Save produced model
model_path = "/Users/boris/Projects/TensorFlowPython/models/"
model_name = "KerasModel"
save_path = saver.save(sess, model_path+model_name+".ckpt")
print "Saved model at ", save_path
graph_path = tf.train.write_graph(sess.graph_def, model_path, model_name+".pb", as_text=True)
print "Saved graph at :", graph_path

# Now freeze the graph (put variables into graph)

input_saver_def_path = ""
input_binary = False
output_node_names = "dense_3_1/Sigmoid"           # Model result node
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = model_path + 'frozen_' + model_name + '.pb'
clear_devices = True


freeze_graph.freeze_graph(graph_path, input_saver_def_path,
                          input_binary, save_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
print "Model is frozen"

# optimizing graph

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ["dense_1_input_1"],        # an array of the input node(s)
    ["dense_3_1/Sigmoid"],      # an array of output nodes
    tf.float32.as_datatype_enum)

# Save the optimized graph

tf.train.write_graph(output_graph_def, model_path, "optimized_" + model_name + ".pb", as_text=False)
tf.train.write_graph(output_graph_def, model_path, "optimized_text_" + model_name + ".pb", as_text=True)


# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
#predictions = model.predict(X)
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)