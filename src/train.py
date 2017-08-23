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

[ ] Activation: tanh, relu, sigmoid, linear
    "relu": nn.Activations.RELU,
    "tanh": nn.Activations.TANH,
    "sigmoid": nn.Activations.SIGMOID,
    "linear": nn.Activations.LINEAR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
#import sys
import tempfile
import matplotlib.pyplot as plt
import matplotlib

INPUT_DIM = 2
NUM_CLASSES = 2
NUM_SAMPLES = 20 # 200


# https://github.com/tensorflow/playground/blob/master/src/dataset.ts
class DataSet:
    DATA_CIRCLE = "circle"
    DATA_XOR = "xor"
    DATA_GAUSS = "gauss"
    DATA_SPIRAL = "spiral"
    DATA_NAMES = [DATA_CIRCLE, DATA_XOR, DATA_GAUSS, DATA_SPIRAL]

    def __init__(self, dataset_name, num_samples, noise):
        self.points = None   # 2d point coordinates (x1, x2)
        self.labels = None   # one-hot class labels
        self.batch_index = 0
        data_gene_func = {
            self.DATA_CIRCLE: self.data_circle,
            self.DATA_XOR: self.data_xor,
            self.DATA_GAUSS: self.data_gauss,
            self.DATA_SPIRAL: self.data_spiral
        }
        data_gene_func[dataset_name](num_samples, noise)

    def data_circle(self, num_samples, noise):
        pass

    def data_xor(self, num_samples, noise):
        pass

    def data_gauss(self, num_samples, noise):
        """
        two Gaussian
        :param num_samples:
        :param noise:
        :return:
        """
        def value_scale(value, domain_min, domain_max, range_min, range_max):
            return range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)

        variance = value_scale(noise, 0, 0.5, 0.5, 4)
        assert (NUM_CLASSES == 2 and INPUT_DIM == 2)  # otherwise this data dimensionality is incorrect
        data_pos = (np.random.multivariate_normal((2, 2), np.identity(INPUT_DIM) * variance, (num_samples // 2)))
        data_neg = (np.random.multivariate_normal((-2, -2), np.identity(INPUT_DIM) * variance, (num_samples // 2)))
        self.points = np.concatenate((data_pos, data_neg))
        # create a one hot array of labels
        self.labels = np.array([np.zeros(len(data_pos)), np.ones(len(data_neg))]).ravel().astype(int)
        self.shuffle_data()

    def shuffle_data(self):
        zipped = list(zip(self.points, self.labels))
        random.shuffle(zipped)
        self.points, self.labels = zip(*zipped)
        self.points = np.array(self.points)
        self.labels = np.array(self.labels).astype(int)
        # print(self.points)
        # print(self.labels)
        # sys.stdout.flush()

    def data_spiral(self, num_samples, noise):
        pass

    def next_batch(self, mini_batch_size):
        while True:
            assert(len(self.points) == len(self.labels))
            points = [self.points[i % len(self.points)] for i in range(self.batch_index,
                                                                       self.batch_index + mini_batch_size)]
            labels = [self.labels[i % len(self.labels)] for i in range(self.batch_index,
                                                                       self.batch_index + mini_batch_size)]
            self.batch_index += mini_batch_size
            return np.array(points), np.array(labels)

class Classifier:
    def __init__(self):
        self.batch_size = 10
        self.session = tf.Session()

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, INPUT_DIM], name='x')  # input data
        self.y = tf.placeholder(tf.uint8, [None], name='y')  # GT classes
        self.y1h = tf.one_hot(self.y, depth=2, name='y1h')  # GT classes one-hot

        self.w = tf.Variable(tf.random_normal([INPUT_DIM, NUM_CLASSES]), name='w-out')
        self.b = tf.Variable(tf.zeros([NUM_CLASSES]), name='b-out')

        self.yp = tf.matmul(self.x, self.w) + self.b  # tf.add(tf.matmul(self.x, self.w), self.b, name='y_p')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y1h, logits=self.yp, name='cross_entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')

        with tf.name_scope('adam_optimizer'):
            # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

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

        for step in range(1000 + 1):
            batch = data.next_batch(self.batch_size)
            self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
            if step % 100 == 0:
                train_loss = self.loss.eval(feed_dict={self.x: batch[0], self.y: batch[1]}, session=self.session)
                print('step %d, training loss: %g' % (step, train_loss))

    def predict_y(self, x):
        yp = self.yp.eval(feed_dict={self.x: x}, session=self.session)
        return np.argmax(yp, axis=1)

if __name__ == '__main__':
    dataset_name = DataSet.DATA_GAUSS
    data = DataSet(dataset_name, 200, 0.5)

    classifier = Classifier()
    classifier.build()
    classifier.train(data)

    # matplotlib.interactive(False)
    # plot the resulting classifier
    colormap = matplotlib.colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
    h = 0.02
    x_min, x_max = -6, 6  # data.points[:, 0].min() - 1, data.points[:, 0].max() + 1
    y_min, y_max = -6, 6  # data.points[:, 1].min() - 1, data.points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = classifier.predict_y(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = plt.figure(figsize=(4, 4), dpi=75)
    plt.contourf(xx, yy, z, cmap=colormap, alpha=0.8)
    point_color = data.labels
    plt.scatter(data.points[:, 0], data.points[:, 1], c=point_color, s=30, cmap=colormap)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    fig.savefig(dataset_name + '.png')
