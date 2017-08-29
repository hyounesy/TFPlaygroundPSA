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
import numpy as np
import random
from config import Config

# https://github.com/tensorflow/playground/blob/master/src/dataset.ts
class DataSet:
    DATA_CIRCLE = "circle"
    DATA_XOR = "xor"
    DATA_GAUSS = "gauss"
    DATA_SPIRAL = "spiral"
    data_names = [DATA_CIRCLE, DATA_XOR, DATA_GAUSS, DATA_SPIRAL]

    def __init__(self, dataset_name, num_samples, noise):
        self.points = None   # 2d point coordinates (x1, x2)
        self.labels = None   # one-hot class labels
        self.batch_index = 0
        self.dataset_name = dataset_name
        self.noise = noise
        data_gen_func = {
            self.DATA_CIRCLE: self.data_circle,
            self.DATA_XOR: self.data_xor,
            self.DATA_GAUSS: self.data_gauss,
            self.DATA_SPIRAL: self.data_spiral
        }
        data_gen_func[dataset_name](num_samples, noise)

    def data_circle(self, num_samples, noise):
        self.points = np.zeros([num_samples, 2])
        self.labels = np.zeros(num_samples).astype(int)
        radius = 5

        def get_circle_label(x, y, xc, yc):
            return 1 if np.sqrt((x - xc)**2 + (y - yc)**2) < (radius * 0.5) else 0

        # Generate positive points inside the circle.
        for i in range(num_samples // 2):
            r = random.uniform(0, radius * 0.5)
            angle = random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = random.uniform(-radius, radius) * noise
            noise_y = random.uniform(-radius, radius) * noise
            self.labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
            self.points[i] = (x, y)


        # Generate negative points outside the circle.
        for i in range(num_samples // 2, num_samples):
            r = random.uniform(radius * 0.7, radius)
            angle = random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = random.uniform(-radius, radius) * noise
            noise_y = random.uniform(-radius, radius) * noise
            self.labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
            self.points[i] = (x, y)

        self.shuffle_data()

    def data_xor(self, num_samples, noise):
        def get_xor_label(px, py):
            return 1 if px * py >= 0 else 0
        self.points = np.zeros([num_samples, 2])
        self.labels = np.zeros(num_samples).astype(int)
        for i in range(num_samples):
            x = random.uniform(-5, 5)
            padding = 0.3
            x += padding if x > 0 else -padding
            y = random.uniform(-5, 5)
            y += padding if y > 0 else -padding
            noise_x = random.uniform(-5, 5) * noise
            noise_y = random.uniform(-5, 5) * noise
            self.labels[i] = get_xor_label(x + noise_x, y + noise_y)
            self.points[i] = (x, y)
        self.shuffle_data()

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
        assert (Config.NUM_CLASSES == 2 and Config.INPUT_DIM == 2)  # otherwise this data dimensionality is incorrect
        data_pos = (np.random.multivariate_normal((2, 2), np.identity(Config.INPUT_DIM) * variance, (num_samples // 2)))
        data_neg = (np.random.multivariate_normal((-2, -2), np.identity(Config.INPUT_DIM) * variance, (num_samples // 2)))
        self.points = np.concatenate((data_pos, data_neg))
        # create a one hot array of labels
        self.labels = np.array([np.zeros(len(data_pos)), np.ones(len(data_neg))]).ravel().astype(int)
        self.shuffle_data()

    def data_spiral(self, num_samples, noise):
        half = num_samples // 2
        self.points = np.zeros([num_samples, 2])
        self.labels = np.zeros(num_samples).astype(int)
        for j in range(num_samples):
            i = j % half
            label = 1
            delta = 0
            if j >= half:  # negative examples
                label = 0
                delta = np.pi
            r = i / half * 5
            t = 1.75 * i / half * 2 * np.pi + delta
            x = r * np.sin(t) + random.uniform(-1, 1) * noise
            y = r * np.cos(t) + random.uniform(-1, 1) * noise
            self.labels[j] = label
            self.points[j] = (x, y)
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

    def num_samples(self):
        if self.points is not None:
            return len(self.points)
        return 0

    def next_training_batch(self, training_ratio, mini_batch_size):
        """
        returns next training batch
        :param mini_batch_size:
        :return:
        """
        num_training = int(self.num_samples() * training_ratio)
        assert(len(self.points) == len(self.labels))
        points = [self.points[i % num_training] for i in range(self.batch_index,
                                                               self.batch_index + mini_batch_size)]
        labels = [self.labels[i % num_training] for i in range(self.batch_index,
                                                               self.batch_index + mini_batch_size)]
        self.batch_index += mini_batch_size
        return np.array(points), np.array(labels).astype(int)

    def get_test(self, training_ratio):
        num_training = int(self.num_samples() * training_ratio)
        return self.points[num_training:], self.labels[num_training:]

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\t'.join(['x1', 'x2', 'label']) + '\n')
            for i in range(self.num_samples()):
                f.write('\t'.join([str(self.points[i][0]), str(self.points[i][1]), str(self.labels[i])])+ '\n')





