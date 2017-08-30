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
        self.training_ratio = 50  # ratio (percentage) of training to test [10, 20, ..., 90]
        self.learning_rate = 0.3
        # self.num_hidden = 1  # number of hidden layers
        # self.num_hidden_neuron = 4  # number of neurons in each hidden layer
        self.neurons_per_layer = [2, 2]  # number of neurons per hidden layer
        self.activation_h = self.ACTIVATION_TANH  # activation function for hidden layers
        self.regularization_type = self.REGULARIZATION_L1
        self.regularization_rate = 0.1
        self.batch_size = 10
        self.save_graph = False
        self.features_ids = [DataSet.FEATURE_X1, DataSet.FEATURE_X2]  # list of selected feature IDs

    def build(self):
        num_selected_features = len(self.features_ids)

        # mask = [i in self.selected_features_ids for i in range(DataSet.NUM_FEATURES)]
        # self.all_features = tf.placeholder(tf.float32, [None, DataSet.NUM_FEATURES], name='all_features')
        # self.features = tf.boolean_mask(self.features, mask, name='selected_features')

        self.features = tf.placeholder(tf.float32, [None, num_selected_features], name='features')
        self.y = tf.placeholder(tf.uint8, [None], name='y')  # GT classes
        self.y1h = tf.one_hot(self.y, depth=2, name='y1h')  # GT classes one-hot

        curr_out = self.features
        curr_dim = num_selected_features

        regularizer = None
        if self.regularization_type == self.REGULARIZATION_L1:
            regularizer = tf.contrib.layers.l1_regularizer(self.regularization_rate)
        elif self.regularization_type == self.REGULARIZATION_L2:
            regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        with tf.variable_scope('layers', regularizer=regularizer):
            num_hidden = len(self.neurons_per_layer)
            for ih in range(num_hidden):
                next_dim = self.neurons_per_layer[ih]
                w_h = tf.Variable(tf.random_normal([curr_dim, next_dim]), name='wh_'+str(ih))
                w_b = tf.Variable(tf.zeros([next_dim]), name='bh_'+str(ih))
                curr_dim = next_dim
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

        # save the network graph
        if self.save_graph:
            graph_location = tempfile.mkdtemp()
            print('Saving graph to: %s' % graph_location)
            train_writer = tf.summary.FileWriter(graph_location)
            train_writer.add_graph(tf.get_default_graph())

        self.session.run(tf.global_variables_initializer())

    def get_selected_features(self, features):
        return np.transpose(np.transpose(features)[np.array(self.features_ids)])

    def predict_labels(self, features):
        yp = self.yp.eval(feed_dict={self.features: self.get_selected_features(features)}, session=self.session)
        return np.argmax(yp, axis=1)

    def train(self, data, restart=True, num_steps=1000):
        """
        Runs the classifier for a certain number of steps
        :param data: the input data (training+test)
        :param restart: whether to restart the classifier (reinitialize weights)
        :param num_steps: number of steps to run the classifier
        :return: train_loss, test_loss
        """
        if restart:
            tf.initialize_all_variables()

        for step in range(num_steps):
            batch_features, batch_labels = data.next_training_batch(self.training_ratio, self.batch_size)
            self.train_step.run(feed_dict={self.features: self.get_selected_features(batch_features),
                                           self.y: batch_labels}, session=self.session)
        # compute test loss
        train_features, train_labels = data.get_training(self.training_ratio)
        train_loss = self.loss.eval(feed_dict={self.features: self.get_selected_features(train_features),
                                               self.y: train_labels}, session=self.session)
        # compute training loss
        test_features, test_labels = data.get_test(self.training_ratio)
        test_loss = self.loss.eval(feed_dict={self.features: self.get_selected_features(test_features),
                                              self.y: test_labels}, session=self.session)
        return train_loss, test_loss
