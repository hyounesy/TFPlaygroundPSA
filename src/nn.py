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
        self.session = None
        self.activation_func = {
            self.ACTIVATION_RELU: tf.nn.relu,
            self.ACTIVATION_TANH: tf.nn.tanh,
            self.ACTIVATION_SIGMOID: tf.nn.sigmoid,
            self.ACTIVATION_LINEAR: lambda x: x,
        }
        self.perc_train = 50  # percentage of training to test [10, 20, ..., 90]
        self.learning_rate = 0.3
        self.neurons_per_layer = [2, 2]  # number of neurons per hidden layer
        self.activation_h = self.ACTIVATION_TANH  # activation function for hidden layers
        self.regularization_type = self.REGULARIZATION_L1
        self.regularization_rate = 0.1
        self.batch_size = 10
        self.save_graph = False
        self.features_ids = [DataSet.FEATURE_X1, DataSet.FEATURE_X2]  # list of selected feature IDs

    def build(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        tf.reset_default_graph()
        self.session = tf.Session()
        num_selected_features = len(self.features_ids)

        # could not figure out how to use the boolean_mask.
        # So for now, the input should only include the selected features
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
            regularizer = tf.contrib.layers.l1_regularizer(scale=self.regularization_rate)
        elif self.regularization_type == self.REGULARIZATION_L2:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization_rate)

        with tf.variable_scope('layers'):
            num_hidden = len(self.neurons_per_layer)
            for ih in range(num_hidden):
                next_dim = self.neurons_per_layer[ih]
                w_h = tf.get_variable(name='wh_'+str(ih), shape=[curr_dim, next_dim],
                                      initializer=tf.random_normal_initializer, regularizer=regularizer)
                w_b = tf.get_variable(name='bh_'+str(ih), shape=[next_dim], initializer=tf.zeros_initializer)
                curr_dim = next_dim
                curr_out = self.activation_func[self.activation_h](tf.matmul(curr_out, w_h) + w_b)

            w_out = tf.get_variable(name='w-out', shape=[curr_dim, Config.NUM_CLASSES],
                                    initializer=tf.random_normal_initializer, regularizer=regularizer)
            b_out = tf.get_variable(name='b-out', shape=[Config.NUM_CLASSES], initializer=tf.zeros_initializer)
            self.yp = (tf.matmul(curr_out, w_out) + b_out)  # predicted y
            # output activation is always tanh: https://github.com/tensorflow/playground/blob/master/src/playground.ts#L956
            activation = tf.nn.tanh(self.yp)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y1h, logits=activation, name='cross_entropy')
            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

            # loss value calculated in tf playground example: https://github.com/tensorflow/playground/blob/master/src/playground.ts#L845
            # note: not adding 0.5 * because here the labels are 0 or 1 vs. in the web version labels are -1 or 1
            self.tf_playground_loss = tf.reduce_mean(
                tf.square(tf.cast(
                    tf.not_equal(tf.argmax(self.yp, 1), tf.argmax(self.y1h, 1)), tf.float32)),
                name='tf_playground_loss')

            self.loss = cross_entropy_loss
            if regularizer:
                self.loss += tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                # tf.contrib.layers.apply_regularization(regularizer)
                # self.loss = self.loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

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
            batch_features, batch_labels = data.next_training_batch(self.perc_train, self.batch_size)
            self.train_step.run(feed_dict={self.features: self.get_selected_features(batch_features),
                                           self.y: batch_labels}, session=self.session)
        # compute test loss
        train_features, train_labels = data.get_training(self.perc_train)
        train_loss = self.tf_playground_loss.eval(feed_dict={self.features: self.get_selected_features(train_features),
                                                  self.y: train_labels}, session=self.session)
        # compute training loss
        test_features, test_labels = data.get_test(self.perc_train)
        test_loss = self.tf_playground_loss.eval(feed_dict={self.features: self.get_selected_features(test_features),
                                                 self.y: test_labels}, session=self.session)
        return train_loss, test_loss
