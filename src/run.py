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
# sys.path.append('/Users/hyounesy/Research/cass2017_vis/src')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
from matplotlib import colors
from dataset import DataSet
from nn import Classifier
from config import Config
import random
import os
import shutil
import time

class Run:
    """
    A single run
    """
    num_samples = 200  # always fixed
    range_noise = list(range(0, 51, 5))
    perc_train_values = list(range(10, 91, 10)) # percentage of training to test
    range_batch_size = [1, 30]
    learning_rates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    regularization_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    range_hidden = [0, 6]
    range_hidden_neuron = [1, 8]

    MODE_FULL = 'full'  # a single directory, with randomized data for each run
    MODE_PSA_RUNS = 'psa_runs'  # a few randomized data, in separate directories

    def __init__(self):
        self.data = None
        self.nn = None

    def randomize_data(self, dataset_name=None, noise=None):
        """
        Build dataset with randomized parameters
        :param dataset_name: dataset name. if None, will randomize
        :param noise: noise [0 .. 50]. if None will randomly pick
        :return: None
        """
        # dataset parameters
        dataset_name = random.choice(DataSet.all_data_names) if dataset_name is None else dataset_name
        noise = random.choice(self.range_noise) if noise is None else noise
        self.data = DataSet(dataset_name, self.num_samples, noise)

    def randomize_training_params(self):
        """
        Creates classifier and network with randomized parameters
        :return: None
        """
        self.nn = Classifier()
        self.nn.perc_train = random.choice(self.perc_train_values)
        self.nn.batch_size = random.randint(*self.range_batch_size)
        self.nn.learning_rate = random.choice(self.learning_rates)
        self.nn.neurons_per_layer = [random.randint(*self.range_hidden_neuron)
                                     for _ in range(random.randint(*self.range_hidden))]
        self.nn.activation_h = random.choice(Classifier.activations_names)
        self.nn.regularization_type = random.choice(Classifier.regularization_names)
        self.nn.regularization_rate = random.choice(self.regularization_rates)
        feature_ids = [i for i in range(DataSet.NUM_FEATURES)]
        random.shuffle(feature_ids)
        feature_ids = feature_ids[0: random.randint(1, len(feature_ids))]
        feature_ids.sort()
        self.nn.features_ids = feature_ids
        self.nn.build()

    PARAM_TYPE_INT = 'int'
    PARAM_TYPE_DOUBLE = 'double'
    PARAM_TYPE_STR = 'string'
    PARAM_TYPE_OUTPUT = 'output'

    @staticmethod
    def param_info_header():
        return ['label', 'name', 'type', 'info']

    def param_info(self):
        max_hidden = self.range_hidden[1]
        return ([['data', 'Data', self.PARAM_TYPE_STR, 'Which dataset do you want to use?'],
                 ['noise', 'Noise', self.PARAM_TYPE_INT, 'Noise'],
                 ['training_ratio', 'Training Ratio', self.PARAM_TYPE_INT, 'Ratio of training to test data'],
                 ['batch_size', 'Batch Size', self.PARAM_TYPE_INT, 'Batch Size']] +
                [[f, f, self.PARAM_TYPE_INT, f] for f in DataSet.feature_idx_to_name] +
                [['layer_count', 'Layers Count', self.PARAM_TYPE_INT, 'Number of hidden layers'],
                 ['neuron_count', 'Neurons Count', self.PARAM_TYPE_INT, 'Total number of neurons in all hidden layers']] +
                [['H'+str(i), 'H'+str(i), self.PARAM_TYPE_INT, 'H'+str(i)] for i in range(1, max_hidden + 1)] +
                [['learning_rate', 'Learning rate', self.PARAM_TYPE_DOUBLE, 'Learning rate'],
                 ['activation', 'Activation', self.PARAM_TYPE_STR, 'Activation'],
                 ['regularization', 'Regularization', self.PARAM_TYPE_STR, 'Regularization'],
                 ['regularization_rate', 'Regularization rate', self.PARAM_TYPE_DOUBLE, 'Regularization rate']])

    def param_names(self):
        """
        returns array of string names for the parameters. matching 1-to-1 with param_str
        :return:
        """
        info = self.param_info()
        return [info[i][0] for i in range(len(info))]

    def param_str(self):
        """
        returns array of parameter values in string format. matching 1-to-1 to the param_names()
        :return:
        """
        layer_count = len(self.nn.neurons_per_layer)
        max_hidden = self.range_hidden[1]
        return ([self.data.dataset_name,
                 str(self.data.noise),
                 str(self.nn.perc_train),
                 str(self.nn.batch_size)] +
                ['1' if i in self.nn.features_ids else '0' for i in DataSet.all_features] +
                [str(layer_count),
                 str(sum(self.nn.neurons_per_layer))] +
                [str(self.nn.neurons_per_layer[i]) if i < layer_count else '0' for i in range(max_hidden)] +
                [str(self.nn.learning_rate),
                 self.nn.activation_h,
                 self.nn.regularization_type,
                 str(self.nn.regularization_rate)])

    def save_plot(self, filename):
        """
        Generates the plot using the current data and training state
        :param filename: output filename
        :return: None
        """
        # matplotlib.interactive(False)
        # plot the resulting classifier
        colormap = colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
        x_min, x_max = -6, 6
        y_min, y_max = -6, 6
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        data_points = np.c_[xx.ravel(), yy.ravel()]
        data_grid = DataSet(None, len(data_points), 0, data_points=data_points)

        z = self.nn.predict_labels(data_grid.features).reshape(xx.shape)
        fig = plt.figure(figsize=(4, 4), dpi=75)
        # plt.imshow(z, cmap=colormap, interpolation='nearest')
        plt.contourf(xx, yy, z, cmap=colormap, alpha=0.8)
        num_training = self.data.num_training(self.nn.perc_train)
        point_color = self.data.labels
        # plot training data points
        plt.scatter(self.data.points[:num_training, 0], self.data.points[:num_training, 1],
                    c=point_color[:num_training], edgecolors='w', s=40, cmap=colormap)
        # plot test data points
        plt.scatter(self.data.points[num_training:, 0], self.data.points[num_training:, 1],
                    c=point_color[num_training:], edgecolors='k', s=30, cmap=colormap)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        fig.savefig(filename)
        plt.close()

    def save_current_run(self, filename):
        yp = self.nn.predict_labels(self.data.features)
        if Config.SAVE_LABELS_NEG_POS:
            yp = [-1 if label == 0 else label for label in yp]
        header = 'label_pred'
        with open(filename, 'w') as f:
            f.write(header + '\n')
            for v in yp:
                f.write(str(v) + '\n')

    def calc_tpr_fpr(self):
        """
        calculates the true positive rate and false positive rate
        :return: tpr, fpr
        """
        num_training = self.data.num_training(self.nn.perc_train)
        y_test = self.data.labels[num_training:] # true labels
        yp_test = self.nn.predict_labels(self.data.features)[num_training:] # predicted labels
        num_p = list(y_test).count(1)  # number of positive labels
        num_tp = [y == 1 and yp == 1 for y, yp in zip(y_test, yp_test)].count(True)  # true positives
        num_n = list(y_test).count(0)  # number of negative labels
        num_fp = [y == 0 and yp == 1 for y, yp in zip(y_test, yp_test)].count(True)  # false positives
        tpr = 0 if num_tp == 0 else num_tp/num_p  # true positive rate
        fpr = 0 if num_fp == 0 else num_fp/num_n  # false positive rate
        return tpr, fpr

    @staticmethod
    def create_dir(dirname, clean=False):
        if clean:
            shutil.rmtree(dirname, ignore_errors=True)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def execute_runs(self, mode, num_runs):
        """
        Executes several training runs, each with different parameters and saves the results
        :param mode: experiment mode.
            MODE_FULL randomizes all parameters including the input data, per run
            MODE_PSA_RUNS generates different datasets and runs the psa separately for each
        :param num_runs: number of runs per experiment
        :return:
        """
        iter_index = -1
        while True:
            iter_index += 1
            if mode == self.MODE_FULL:
                if iter_index == 1:
                    break
                out_dir = '../output/full'
                self.create_dir(out_dir, clean=True)
                curr_data = None
            elif mode == self.MODE_PSA_RUNS:
                if iter_index >= len(DataSet.all_data_names):
                    break
                noise = 25
                dataset_name = DataSet.all_data_names[iter_index]
                out_dir = '../output/' + dataset_name + '_' + str(noise)
                self.create_dir(out_dir, clean=True)
                curr_data = DataSet(dataset_name, num_samples=Run.num_samples, noise=noise)
                curr_data.save_to_file(out_dir + '/input.txt')
            else:
                print("Invalid mode:" + str(mode))
                return

            # create write the header for the runs.txt file
            f_runs = open(out_dir + '/index.txt', 'w')
            all_param_info = ([['ID', 'ID', self.PARAM_TYPE_OUTPUT, 'ID'],
                               ['imagePath', 'Image path', self.PARAM_TYPE_OUTPUT, 'Output image path']] +
                              self.param_info() +
                              [['epoch', 'Epoch', self.PARAM_TYPE_INT, 'Epoch'],
                               ['total_time', 'Total time (ms)', self.PARAM_TYPE_OUTPUT, 'Total time at this epoch'],
                               ['mean_time', 'Mean time (ms)', self.PARAM_TYPE_OUTPUT, 'Mean time per epoch'],
                               ['train_loss', 'Training loss', self.PARAM_TYPE_OUTPUT, 'Training loss at step'],
                               ['test_loss', 'Test loss', self.PARAM_TYPE_OUTPUT, 'Test loss at step'],
                               ['TPR', 'True Positive Rate', self.PARAM_TYPE_OUTPUT, 'True Positive Rate'],
                               ['FPR', 'False Positive Rate', self.PARAM_TYPE_OUTPUT, 'False Positive Rate']
                               ])

            # save the paramInfo.txt
            with open(out_dir + '/paramInfo.txt', 'w') as fpi:
                fpi.write('\t'.join(self.param_info_header()) + '\n')
                fpi.write('\n'.join(['\t'.join(i) for i in all_param_info]))

            # write the header for the runs.txt
            f_runs.write('\t'.join([i[0] for i in all_param_info]) + '\n')
            images_dir = out_dir + '/images'
            runs_dir = out_dir + '/runs'
            self.create_dir(images_dir, clean=True)
            self.create_dir(runs_dir, clean=True)

            row_index = 0
            for i in range(num_runs):
                if curr_data is None:
                    self.randomize_data()  # randomize the data every time
                else:
                    self.data = curr_data  # reuse the same data
                self.randomize_training_params()
                print('  '.join(a[0] + ': ' + a[1] for a in zip(self.param_names(), self.param_str())))

                prev_step = 0
                total_time = 0
                for epoch in [100, 200, 400, 800, 1600]:
                    # curr_step = int(epoch * self.data.num_samples() / self.nn.batch_size)
                    curr_step = epoch # in the online demo epoch == iter: https://github.com/tensorflow/playground/blob/67cf64ffe1fc53967d1c979d26d30a4625d18310/src/playground.ts#L898

                    time_start = time.time()
                    test_loss, train_loss = self.nn.train(self.data, restart=False, num_steps=curr_step - prev_step)
                    total_time += (time.time() - time_start) * 1000.0
                    mean_time = total_time / epoch

                    tpr, fpr = self.calc_tpr_fpr()

                    print('epoch %d (step %d), total_time: %g, mean_time: %g, '
                          'training loss: %g, test loss: %g, tpr: %g, fpr: %g' %
                          (epoch, curr_step, total_time, mean_time,
                           train_loss, test_loss, tpr, fpr))

                    image_filename = images_dir + '/' + str(row_index) + ".png"
                    run_filename = runs_dir + '/' + str(row_index) + ".txt"

                    f_runs.write('\t'.join(
                        [str(row_index),
                         image_filename] +
                        self.param_str() +
                        [str(epoch),
                         str(total_time),
                         str(mean_time),
                         str(train_loss),
                         str(test_loss),
                         str(tpr),
                         str(fpr)]) +
                                 '\n')
                    row_index += 1
                    self.save_plot(image_filename)
                    self.save_current_run(run_filename)
                    prev_step = curr_step


if __name__ == '__main__':
    run = Run()
    run.execute_runs(run.MODE_FULL, 40)


"""
make two datasets: one flattened
TODO:
for mode=full, pregenerate several datasets with different noise levels 0, 10, 20, 30, 40, 50

VisRseq TODO:
index.txt -> runs.txt
checkbox to enable/disable sorting by impact?
keep the aspect ratio of the image
mds is good. sum up/down bad. how to get rid of it?
adds quotes, without them won't work
"""
