# Copyright 2017 Hamid Younesy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
TODO:
[ ] training + test

[ ] loss

[ ] learning rate: 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10

[ ] regularization: None, L1, L2:
    "none": null,
    "L1": nn.RegularizationFunction.L1,
    "L2": nn.RegularizationFunction.L2

[ ] regularization rate: 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10

[ ] Activation nad outputActivation: tf.nn.tanh, tf.nn.relu, tf.nn.sigmoid, linear (none)
    "relu": nn.Activations.RELU,
    "tanh": nn.Activations.TANH,
    "sigmoid": nn.Activations.SIGMOID,
    "linear": nn.Activations.LINEAR

[ ] outputActivation = nn.Activations.TANH
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from dataset import DataSet
from config import Config

batch_size = 30
dataset_name = DataSet.DATA_CIRCLE
num_samples = 100
noise = 0.0


class Classifier:
    ACTIVATION_RELU = 'ReLU'
    ACTIVATION_TANH = 'Tanh'
    ACTIVATION_SIGMOID = 'Sigmoid'
    ACTIVATION_LINEAR = 'Linear'
    activations_names = [ACTIVATION_RELU, ACTIVATION_TANH, ACTIVATION_SIGMOID, ACTIVATION_LINEAR]

    REGULARIZATION_NONE = 'None'
    REGULARIZATION_L1 = 'L1'
    REGULARIZATION_L2 = 'L2'
    regularization_names = [REGULARIZATION_NONE, REGULARIZATION_L1, REGULARIZATION_L2]

    def __init__(self):
        self.session = tf.Session()
        self.activation_func = {
            self.ACTIVATION_RELU: tf.nn.relu,
            self.ACTIVATION_TANH: tf.nn.tanh,
            self.ACTIVATION_SIGMOID: tf.nn.sigmoid,
            self.ACTIVATION_LINEAR: lambda x: x,
        }
        self.learning_rate = 0.3
        self.num_hidden = 1  # number of hidden layers
        self.num_hidden_neuron = 4  # number of neurons in each hidden layer
        self.activation_h = self.ACTIVATION_TANH  # activaion function for hidden layers
        self.regularization_type = self.REGULARIZATION_L1
        self.regularization_rate = 0.1

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, Config.INPUT_DIM], name='x')  # input data
        self.y = tf.placeholder(tf.uint8, [None], name='y')  # GT classes
        self.y1h = tf.one_hot(self.y, depth=2, name='y1h')  # GT classes one-hot

        curr_out = self.x
        curr_dim = Config.INPUT_DIM

        regularizer = None
        if self.regularization_type == self.REGULARIZATION_L1:
            regularizer = tf.contrib.layers.l1_regularizer(self.regularization_rate)
        elif self.regularization_type == self.REGULARIZATION_L2:
            regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        with tf.variable_scope('layers', regularizer=regularizer):
            for ih in range(self.num_hidden):
                w_h = tf.Variable(tf.random_normal([curr_dim, self.num_hidden_neuron]), name='wh_'+str(ih))
                w_b = tf.Variable(tf.zeros([self.num_hidden_neuron]), name='bh_'+str(ih))
                curr_dim = self.num_hidden_neuron
                curr_out = self.activation_func[self.activation_h](tf.matmul(curr_out, w_h) + w_b)

                w_out = tf.Variable(tf.random_normal([curr_dim, Config.NUM_CLASSES]), name='w-out')
                b_out = tf.Variable(tf.zeros([Config.NUM_CLASSES]), name='b-out')
                self.yp = (tf.matmul(curr_out, w_out) + b_out)  # predicted y
                activation = tf.nn.tanh(self.yp)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y1h, logits=activation, name='cross_entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            if regularizer:
                self.loss = self.loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope('optimizer'):
            # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.yp, 1), tf.argmax(self.y1h, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

    def train(self, data):
        # sess = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.global_variables_initializer())

        test_data = data.get_test()
        for step in range(1000 + 1):
            batch = data.next_training_batch(batch_size)
            self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
            if step % 100 == 0:
                train_loss = self.loss.eval(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
                test_loss = self.loss.eval(feed_dict={self.x: test_data[0], self.y: test_data[1]}, session=self.session)
                print('step %d, training loss: %g, test loss: %g' % (step, train_loss, test_loss))

    def predict_y(self, x):
        yp = self.yp.eval(feed_dict={self.x: x}, session=self.session)
        return np.argmax(yp, axis=1)

if __name__ == '__main__':

    data = DataSet(dataset_name, num_samples, noise)

    classifier = Classifier()
    classifier.build()
    classifier.train(data)

    # matplotlib.interactive(False)
    # plot the resulting classifier
    colormap = colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
    x_min, x_max = -6, 6  # data.points[:, 0].min() - 1, data.points[:, 0].max() + 1
    y_min, y_max = -6, 6  # data.points[:, 1].min() - 1, data.points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    z = classifier.predict_y(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = plt.figure(figsize=(4, 4), dpi=75)
    #plt.imshow(z, cmap=colormap, interpolation='nearest')
    plt.contourf(xx, yy, z, cmap=colormap, alpha=0.8)
    point_color = data.labels
    plt.scatter(data.points[:, 0], data.points[:, 1], c=point_color, s=30, cmap=colormap)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fig.savefig(dataset_name + '.png')
