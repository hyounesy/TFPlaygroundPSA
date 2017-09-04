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


class DataSet:
    """
    Generate the four training datasets similar to the ones in the playground online demo:
    https://github.com/tensorflow/playground/blob/master/src/dataset.ts
    Note that the labels are internally generated as 0, 1 (as opposed to the -1, 1 in the online demo)
    however can be saved as -1, 1 when Config.SAVE_LABELS_NEG_POS==True
    """

    DATA_CIRCLE = "circle"
    DATA_XOR = "xor"
    DATA_GAUSS = "gauss"
    DATA_SPIRAL = "spiral"
    all_data_names = [DATA_CIRCLE, DATA_XOR, DATA_GAUSS, DATA_SPIRAL]

    FEATURE_X1 = 0
    FEATURE_X2 = 1
    FEATURE_X1SQ = 2
    FEATURE_X2SQ = 3
    FEATURE_X1X2 = 4
    FEATURE_SIN_X1 = 5
    FEATURE_SIN_X2 = 6
    NUM_FEATURES = 7
    all_features = [i for i in range(NUM_FEATURES)]
    feature_idx_to_name = ['X1', 'X2', 'X1Squared', 'X2Squared', 'X1X2', 'sinX1', 'sinX2']
    #feature_name_to_idx = {feature_idx_to_name[i]: i for i in range(NUM_FEATURES)}

    def __init__(self, dataset_name, num_samples, noise, data_points=None, data_labels=None):
        self.points = None
        self.labels = None  # class labels
        self.features = None  # post-processed features
        self.batch_index = 0
        self.dataset_name = dataset_name
        self.noise = noise

        if dataset_name is not None:
            assert (Config.NUM_CLASSES == 2 and Config.POINTS_DIM == 2)  # otherwise these generators won't work
            data_gen_func = {
                self.DATA_CIRCLE: self.data_circle,
                self.DATA_XOR: self.data_xor,
                self.DATA_GAUSS: self.data_gauss,
                self.DATA_SPIRAL: self.data_spiral
            }
            points, labels = data_gen_func[dataset_name](num_samples, noise)
            self.points, self.labels = self.shuffle_points(points, labels)
            self.features = self.create_features(self.points)
        elif data_points is not None:
            self.points = data_points
            self.features = self.create_features(self.points)
            self.labels = data_labels

    @staticmethod
    def create_from_file(filename):
        input_table = np.genfromtxt(filename, dtype=None, delimiter='\t', names=True, autostrip=False)
        points = np.transpose(np.array([input_table['X1'], input_table['X2']]))
        labels = input_table['label']
        for i in range(len(labels)):
            if Config.SAVE_LABELS_NEG_POS and labels[i] == -1:
                labels[i] = 0
        return DataSet(None, num_samples=len(points), noise=0, data_points=points, data_labels=labels)

    @staticmethod
    def data_circle(num_samples, noise):
        """
        Generates the two circles dataset with the given number of samples and noise
        :param num_samples: total number of samples
        :param noise: noise percentage (0 .. 50)
        :return: None
        """
        radius = 5

        def get_circle_label(x, y, xc, yc):
            return 1 if np.sqrt((x - xc)**2 + (y - yc)**2) < (radius * 0.5) else 0

        noise *= 0.01
        points = np.zeros([num_samples, 2])
        labels = np.zeros(num_samples).astype(int)
        # Generate positive points inside the circle.
        for i in range(num_samples // 2):
            r = random.uniform(0, radius * 0.5)
            angle = random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = random.uniform(-radius, radius) * noise
            noise_y = random.uniform(-radius, radius) * noise
            labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
            points[i] = (x, y)
        # Generate negative points outside the circle.
        for i in range(num_samples // 2, num_samples):
            r = random.uniform(radius * 0.7, radius)
            angle = random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = random.uniform(-radius, radius) * noise
            noise_y = random.uniform(-radius, radius) * noise
            labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
            points[i] = (x, y)
        return points, labels

    @staticmethod
    def data_xor(num_samples, noise):
        """
        Generates the xor (checker) dataset with the given number of samples and noise
        :param num_samples: total number of samples
        :param noise: noise percentage (0 .. 50)
        :return: None
        """
        def get_xor_label(px, py):
            return 1 if px * py >= 0 else 0

        noise *= 0.01
        points = np.zeros([num_samples, 2])
        labels = np.zeros(num_samples).astype(int)
        for i in range(num_samples):
            x = random.uniform(-5, 5)
            padding = 0.3
            x += padding if x > 0 else -padding
            y = random.uniform(-5, 5)
            y += padding if y > 0 else -padding
            noise_x = random.uniform(-5, 5) * noise
            noise_y = random.uniform(-5, 5) * noise
            labels[i] = get_xor_label(x + noise_x, y + noise_y)
            points[i] = (x, y)
        return points, labels

    @staticmethod
    def data_gauss(num_samples, noise):
        """
        Generates the two gaussian dataset with the given number of samples and noise
        :param num_samples: total number of samples
        :param noise: noise percentage (0 .. 50)
        :return: None
        """
        def value_scale(value, domain_min, domain_max, range_min, range_max):
            return range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)

        noise *= 0.01
        variance = value_scale(noise, 0, 0.5, 0.5, 4)
        data_pos = (np.random.multivariate_normal((-2, -2), np.identity(Config.POINTS_DIM) * variance, (num_samples // 2)))
        data_neg = (np.random.multivariate_normal((2, 2), np.identity(Config.POINTS_DIM) * variance, (num_samples // 2)))
        points = np.concatenate((data_pos, data_neg))
        # create the array of labels
        labels = np.array([np.zeros(len(data_pos)), np.ones(len(data_neg))]).ravel().astype(int)
        return points, labels

    @staticmethod
    def data_spiral(num_samples, noise):
        """
        Generates the spiral dataset with the given number of samples and noise
        :param num_samples: total number of samples
        :param noise: noise percentage (0 .. 50)
        :return: None
        """
        noise *= 0.01
        half = num_samples // 2
        points = np.zeros([num_samples, 2])
        labels = np.zeros(num_samples).astype(int)
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
            labels[j] = label
            points[j] = (x, y)
        return points, labels

    @staticmethod
    def shuffle_points(points, labels):
        """
        shuffle two lists of points and labels, while keeping them matched
        :param points: list of points
        :param labels: list of labels
        :return: shuffled_points, shuffled_labels
        """
        zipped = list(zip(points, labels))
        random.shuffle(zipped)
        shuffled_points, shuffled_labels = zip(*zipped)
        # print(shuffled_points)
        # print(shuffled_labels)
        # sys.stdout.flush()
        return np.array(shuffled_points), np.array(shuffled_labels).astype(int)
        
    @staticmethod
    def create_features(points):
        """
        Create the features from the points
        :param points: points as an array of [(x1, x2), ...] to create features from
        :return: np.array of features" [(x1, x2, ...), ...]
        """
        x1, x2 = np.transpose(points)
        features = [None] * DataSet.NUM_FEATURES
        features[DataSet.FEATURE_X1] = x1
        features[DataSet.FEATURE_X2] = x2
        features[DataSet.FEATURE_X1SQ] = x1**2
        features[DataSet.FEATURE_X2SQ] = x2**2
        features[DataSet.FEATURE_X1X2] = x1*x2
        features[DataSet.FEATURE_SIN_X1] = np.sin(x1)
        features[DataSet.FEATURE_SIN_X2] = np.sin(x2)
        return np.transpose(features)

    def num_samples(self):
        """
        Returns total number of samples
        :return: total number of samples (training+test)
        """
        if self.features is not None:
            return len(self.features)
        return 0

    def num_training(self, perc_train):
        """
        Returns the number of training data
        :param perc_train: percentage of the training data
        :return:
        """
        return int(self.num_samples() * perc_train / 100)

    def next_training_batch(self, perc_train, mini_batch_size):
        """
        returns next training batch
        :param perc_train: percentage of training to test data. integer between 10 to 90
        :param mini_batch_size:
        :return:
        """
        num_training = self.num_training(perc_train)
        assert(len(self.features) == len(self.labels))
        features = [self.features[i % num_training] for i in range(self.batch_index,
                                                                   self.batch_index + mini_batch_size)]
        labels = [self.labels[i % num_training] for i in range(self.batch_index,
                                                               self.batch_index + mini_batch_size)]
        self.batch_index += mini_batch_size
        return np.array(features), np.array(labels).astype(int)

    def get_training(self, perc_train):
        """
        Returns the training portion of the data based on perc_train
        :param perc_train: percentage of training to test data. integer between 10 to 90
        :return: points[], labels[]
        """
        num_training = self.num_training(perc_train)
        return self.features[:num_training], self.labels[:num_training]

    def get_test(self, perc_train):
        """
        Returns the test portion of the data based on perc_train
        :param perc_train: percentage of training to test data. integer between 10 to 90
        :return: points[], labels[]
        """
        num_training = self.num_training(perc_train)
        return self.features[num_training:], self.labels[num_training:]

    def save_to_file(self, filename, features=all_features, more_columns_header=None, more_columns_rows=None):
        """
        Saves the data table to the file.
        :param filename: output filename
        :param features: features to include in the output
        :param more_columns_header: additional column names to be added to the file. list of string column names.
        :param more_columns_rows: additional columns to be save to the file. should be a list of strings, one per row.
        :return: None
        """
        with open(filename, 'w') as f:
            column_names = ['pid']
            for ft in features:
                column_names.append(self.feature_idx_to_name[ft])
            column_names.append('label')
            header_str = '\t'.join(column_names)
            if more_columns_header is not None:
                header_str += '\t' + more_columns_header
            f.write(header_str + '\n')

            if more_columns_rows is not None:
                assert len(more_columns_rows) == self.num_samples()

            for i in range(self.num_samples()):
                label = self.labels[i]
                if Config.SAVE_LABELS_NEG_POS and label == 0:
                    label = -1
                row_str = [str(i)]
                for ft in features:
                    row_str.append(str(self.features[i][ft]))
                row_str.append(str(label))
                row_str = '\t'.join(row_str)
                if more_columns_rows is not None:
                    row_str += '\t' + more_columns_rows[i]
                f.write(row_str + '\n')


if __name__ == '__main__':
    for dataset_name in DataSet.all_data_names:
        data = DataSet(dataset_name, 200, 25)
        data.save_to_file(dataset_name+'.txt')



