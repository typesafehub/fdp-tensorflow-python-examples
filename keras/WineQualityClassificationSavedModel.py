# based on the example http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# additionally, thread https://github.com/tensorflow/serving/issues/310#issuecomment-297015251 has a lot of info and explanations

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def

# create model 11 inputs -> [4 hidden nodes] -> Y.max()+1 outputs
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(Y.max() + 1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create TF session and set it in Keras
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)

# load dataset
dataframe = pandas.read_csv("../data/winequality_red.csv", delimiter=";", header=None)
dataset = dataframe.values
X = dataset[:,0:11].astype(float)
Y = dataset[:,11].astype(int)

# hot encoding of output https://en.wikipedia.org/wiki/One-hot
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(Y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(Y.max() + 1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, dummy_y, epochs=150, batch_size=10)

print('Done training!')

print "input", model.input.name
print "output", model.output.name

print("Model's parameters")
print(model.get_weights())

# evaluate the model
scores = model.evaluate(X, dummy_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#export_version =...  # version number (integer)
export_dir = "/Users/boris/Projects/TensorFlowPython/savedmodels/WineQuality/"
builder = saved_model_builder.SavedModelBuilder(export_dir)
signature = predict_signature_def(inputs={'winedata': model.input},
                                  outputs={'quality': model.output})
builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
builder.save()
print('Done exporting!')
