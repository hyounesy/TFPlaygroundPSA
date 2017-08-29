# ==============================================================================
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tempfile
from dataset import DataSet
from config import Config

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
        self.training_ratio = 0.5  # ratio of training to test
        self.learning_rate = 0.3
        self.num_hidden = 1  # number of hidden layers
        self.num_hidden_neuron = 4  # number of neurons in each hidden layer
        self.activation_h = self.ACTIVATION_TANH  # activaion function for hidden layers
        self.regularization_type = self.REGULARIZATION_L1
        self.regularization_rate = 0.1
        self.batch_size = 10
        self.save_graph = False

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

        if self.save_graph:
            graph_location = tempfile.mkdtemp()
            print('Saving graph to: %s' % graph_location)
            train_writer = tf.summary.FileWriter(graph_location)
            train_writer.add_graph(tf.get_default_graph())

    def train(self, data, max_steps = 1001, stat_steps=None):
        self.session.run(tf.global_variables_initializer())
        test_data = data.get_test(self.training_ratio)
        stats = []
        for step in range(max_steps):
            batch = data.next_training_batch(self.training_ratio, self.batch_size)
            self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
            # if step % 100 == 0:
            if stat_steps is not None and step in stat_steps:
                train_loss = self.loss.eval(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
                test_loss = self.loss.eval(feed_dict={self.x: test_data[0], self.y: test_data[1]}, session=self.session)
                stats.append({'step': step, 'train_loss': train_loss, 'test_loss': test_loss})
                #print('step %d, training loss: %g, test loss: %g' % (step, train_loss, test_loss))
        return stats

    def predict_y(self, x):
        yp = self.yp.eval(feed_dict={self.x: x}, session=self.session)
        return np.argmax(yp, axis=1)
