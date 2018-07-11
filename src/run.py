# ==============================================================================
# Copyright 2018 Hamid Younesy. All Rights Reserved.
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
# sys.path.append('/Users/hyounesy/Research/TFPlaygroundPSA/src')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from dataset import DataSet
from nn import Classifier
from config import Config
import random
import os
import shutil
import time
matplotlib.use('Agg')


class Run:
    """
    A single run
    """

    num_samples = 200  # always a fixed number of data points
    noise_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    perc_train_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # percentage of training to test
    range_batch_size = [1, 30]
    learning_rates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    regularization_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    range_hidden = [0, 6]
    range_hidden_neuron = [1, 8]
    epochs_per_config = [50, 100, 200, 400]  # number of epochs to run each nnet configuration for
    activations_names = Classifier.activations_names
    regularization_names = Classifier.regularization_names
    fixed_feature_ids = None

    # for debug:
    """
    num_samples = 200  # always a fixed number of data points
    noise_values = [25]
    perc_train_values = [50]  # percentage of training to test
    range_batch_size = [10, 11]
    learning_rates = [0.1]
    regularization_rates = [3.0]
    range_hidden = [3, 3]
    range_hidden_neuron = [4, 4]
    epochs_per_config = [400]  # number of epochs to run each nnet configuration for
    activations_names = [Classifier.ACTIVATION_TANH]
    regularization_names = [Classifier.REGULARIZATION_NONE]
    fixed_feature_ids = [DataSet.FEATURE_X1SQ, DataSet.FEATURE_X2SQ, DataSet.FEATURE_SIN_X1, DataSet.FEATURE_SIN_X2]
    """

    PARAM_TYPE_INT = 'int'
    PARAM_TYPE_DOUBLE = 'double'
    PARAM_TYPE_STR = 'string'
    PARAM_TYPE_OUTPUT = 'output'

    MODE_FULL = 'full'  # a single directory, with randomized data for each run
    MODE_PSA_RUNS = 'psa_runs'  # a few randomized data, in separate directories

    fixed_noise = 25

    # mode_psa_datasets = DataSet.all_data_names
    mode_psa_datasets = [DataSet.DATA_SPIRAL, DataSet.DATA_XOR, DataSet.DATA_CIRCLE, DataSet.DATA_GAUSS]  # debug

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
        noise = random.choice(self.noise_values) if noise is None else noise
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
        self.nn.activation_h = random.choice(self.activations_names)
        self.nn.regularization_type = random.choice(self.regularization_names)
        self.nn.regularization_rate = random.choice(self.regularization_rates)

        # select which input features to use
        if self.fixed_feature_ids is not None:
            self.nn.features_ids = self.feature_ids
        else:
            # random
            feature_bits = random.randint(0, pow(2, DataSet.NUM_FEATURES))
            self.nn.features_ids = [i for i in range(DataSet.NUM_FEATURES) if feature_bits & pow(2, i) != 0]

        self.nn.build()

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
                 ['neuron_count', 'Neurons Count', self.PARAM_TYPE_INT, 'Total number of neurons in hidden layers']] +
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
        x_min, x_max = -6, 6  # grid x bounds
        y_min, y_max = -6, 6  # grid y bounds
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        data_points = np.c_[xx.ravel(), yy.ravel()]
        data_grid = DataSet(None, len(data_points), 0, data_points=data_points)

        try:
            z = self.nn.predict_labels(data_grid.features).reshape(xx.shape)
        except:
            z = np.zeros(np.shape(xx))
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

    @staticmethod
    def create_dir(dirname, clean=False):
        """
        Creates the directory if doesn't exist
        :param dirname: directory path
        :param clean: whether to clean the directory
        :return: None
        """
        if clean:
            shutil.rmtree(dirname, ignore_errors=True)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def save_current_run(self, filename):
        try:
            yp = self.nn.predict_labels(self.data.features)
        except:
            yp = 1 - self.data.labels
        if Config.SAVE_LABELS_NEG_POS:
            yp = [-1 if label == 0 else 1 for label in yp]
        header = 'label_pred'
        with open(filename, 'w') as f:
            f.write(header + '\n')
            for v in yp:
                f.write(str(v) + '\n')

    def calc_tpr_fpr(self):
        """
        calculates the true positive rate and false positive rate
        :return: [train_tpr, train_fpr, test_tpr, test_fpr]
        """
        labels_pred = self.nn.predict_labels(self.data.features)
        num_training = self.data.num_training(self.nn.perc_train)
        stats = []
        for population in ['train', 'test']:
            if population == 'train':
                y = self.data.labels[:num_training]  # true labels for training
                yp = labels_pred[:num_training]  # predicted labels for training
            else:  # population == 'test'
                y = self.data.labels[num_training:]  # true labels for test
                yp = labels_pred[num_training:]  # predicted labels for test

            num_p = list(y).count(1)  # number of positive labels
            num_n = list(y).count(0)  # number of negative labels
            num_tp = [l == 1 and lp == 1 for l, lp in zip(y, yp)].count(True)  # true positives
            num_fp = [l == 0 and lp == 1 for l, lp in zip(y, yp)].count(True)  # false positives
            # num_tn = [l == 0 and lp == 0 for l, lp in zip(y, yp)].count(True)  # true positives
            # num_fn = [l == 1 and lp == 0 for l, lp in zip(y, yp)].count(True)  # true positives
            tpr = 0 if num_tp == 0 else num_tp/num_p  # true positive rate
            fpr = 0 if num_fp == 0 else num_fp/num_n  # false positive rate
            # tnr = 0 if num_tn == 0 else num_tn/num_n  # true negative rate
            # fnr = 0 if num_fn == 0 else num_fn/num_p  # false negative rate
            stats = stats + [tpr, fpr]
        return stats

    def execute_runs(self, mode, num_runs, resume=False):
        """
        Executes several training runs, each with different parameters and saves the results
        :param mode: experiment mode.
            MODE_FULL randomizes all parameters including the input data, per run
            MODE_PSA_RUNS generates different datasets and runs the psa separately for each
        :param num_runs: number of runs per experiment to add to the output
        :param resume: whether to resume the runs. if True, the runs will continue until there are num_runs records.
        :return:
        """
        iter_index = -1
        while True:
            iter_index += 1
            if mode == self.MODE_FULL:
                if iter_index == 1:
                    break
                out_dir = '../output/full'
                self.create_dir(out_dir, clean=not resume)
                curr_data = None
            elif mode == self.MODE_PSA_RUNS:
                if iter_index >= len(self.mode_psa_datasets):
                    break
                noise = self.fixed_noise
                dataset_name = self.mode_psa_datasets[iter_index]
                out_dir = '../output/' + dataset_name + '_' + str(noise)
                self.create_dir(out_dir, clean=not resume)
                input_filename = out_dir + '/input.txt'
                if resume and os.path.exists(input_filename):
                    curr_data = DataSet.create_from_file(input_filename)
                    curr_data.noise = noise
                    curr_data.dataset_name = dataset_name
                    assert(curr_data.num_samples() == Run.num_samples)
                else:
                    curr_data = DataSet(dataset_name, num_samples=Run.num_samples, noise=noise)
                    curr_data.save_to_file(input_filename)
            else:
                print("Invalid mode:" + str(mode))
                return

            run_id = 0
            index_filename = out_dir + '/runsInfo.txt'
            print('index table: ' + index_filename)
            if resume and os.path.exists(index_filename):
                index_table = np.genfromtxt(index_filename, dtype=None, delimiter='\t', names=True, autostrip=False)
                if len(index_table) > 0 and 'ID' in index_table.dtype.fields:
                    run_id = index_table['ID'][-1] + 1
                print('Resuming from ID {}'.format(run_id))

            write_header = (not os.path.exists(index_filename)) or (not resume)
            # create write the header for the runs.txt file
            f_runs = open(index_filename, 'a+' if resume else 'w+')
            all_param_info = \
                ([['ID', 'ID', self.PARAM_TYPE_OUTPUT, 'ID'],
                  ['imagePath', 'Image path', self.PARAM_TYPE_OUTPUT, 'Output image path']] +
                 self.param_info() +
                 [['epoch', 'Epoch', self.PARAM_TYPE_INT, 'Number of Epochs (of processing all training data)'],
                  ['iteration', 'Iterations', self.PARAM_TYPE_INT, 'Number of Iterations (of processing a batch)'],
                  ['success', 'Success', self.PARAM_TYPE_OUTPUT, 'Whether the training finished successfully'],
                  ['total_time', 'Total time (ms)', self.PARAM_TYPE_OUTPUT, 'Total time at this epoch'],
                  ['mean_time', 'Mean time (ms)', self.PARAM_TYPE_OUTPUT, 'Mean time per epoch'],
                  ['train_loss', 'Training loss', self.PARAM_TYPE_OUTPUT, 'Training loss at epoch'],
                  ['test_loss', 'Test loss', self.PARAM_TYPE_OUTPUT, 'Test loss at epoch'],
                  ['train_TPR', 'TPR for train', self.PARAM_TYPE_OUTPUT, 'True positive rate for training data'],
                  ['train_FPR', 'FPR for train', self.PARAM_TYPE_OUTPUT, 'False positive rate for training data'],
                  # ['train_TNR', 'TNR for train', self.PARAM_TYPE_OUTPUT, 'True negative rate for training data'],
                  # ['train_FNR', 'FNR for train', self.PARAM_TYPE_OUTPUT, 'False negative Rate for training data'],
                  ['test_TPR', 'TPR for test', self.PARAM_TYPE_OUTPUT, 'True positive rate for test data'],
                  ['test_FPR', 'FPR for test', self.PARAM_TYPE_OUTPUT, 'False positive rate for test data'],
                  # ['test_TNR', 'TNR for test', self.PARAM_TYPE_OUTPUT, 'True negative rate for test data'],
                  # ['test_FNR', 'FNR for test', self.PARAM_TYPE_OUTPUT, 'False negative Rate for test data'],
                  ])

            # save the paramInfo.txt
            with open(out_dir + '/paramInfo.txt', 'w') as fpi:
                fpi.write('\t'.join(self.param_info_header()) + '\n')
                fpi.write('\n'.join(['\t'.join(i) for i in all_param_info]))

            # write the header for the runs.txt
            if write_header:
                f_runs.write('\t'.join([i[0] for i in all_param_info]) + '\n')
                f_runs.flush()
            images_dir = out_dir + '/images'
            runs_dir = out_dir + '/runs'

            self.create_dir(images_dir, clean=not resume)
            self.create_dir(runs_dir, clean=not resume)

            while run_id < num_runs:
                if curr_data is None:
                    self.randomize_data()  # randomize the data every time
                else:
                    self.data = curr_data  # reuse the same data
                self.randomize_training_params()
                # print the parameters
                print('configuration (%d of %d)' % (int(run_id / len(self.epochs_per_config)) + 1,
                                                    int(num_runs / len(self.epochs_per_config))))
                print(', '.join(a[0] + ': ' + a[1] for a in zip(self.param_names(), self.param_str())))

                prev_step = 0
                total_time = 0
                for epoch in self.epochs_per_config:
                    curr_step = int(epoch * self.data.num_samples() / self.nn.batch_size)
                    # curr_step = epoch # in the online demo epoch == iter: https://github.com/tensorflow/playground/blob/67cf64ffe1fc53967d1c979d26d30a4625d18310/src/playground.ts#L898

                    time_start = time.time()

                    # train the network
                    success = True
                    try:
                        train_loss, test_loss = self.nn.train(self.data, restart=False, num_steps=curr_step - prev_step)
                    except:
                        train_loss, test_loss = 1, 1
                        success = False

                    total_time += (time.time() - time_start) * 1000.0
                    mean_time = total_time / epoch

                    try:
                        train_tpr, train_fpr, test_tpr, test_fpr = self.calc_tpr_fpr()
                    except:
                        train_tpr, train_fpr, test_tpr, test_fpr = 0, 1, 0, 1
                        success = False

                    print('(epoch: %d, step: %d), '
                          '(total_time: %g, mean_time: %g), '
                          '(training loss: %g, test loss: %g), '
                          '(train_tpr: %g, train_fpr: %g test_tpr: %g, test_fpr: %g)' %
                          (epoch, curr_step,
                           round(total_time, 2), round(mean_time, 2),
                           round(train_loss, 2), round(test_loss, 2),
                           round(train_tpr, 2), round(train_fpr, 2), round(test_tpr, 2), round(test_fpr, 2)))

                    image_filename = images_dir + '/' + str(run_id) + ".png"
                    run_filename = runs_dir + '/' + str(run_id) + ".txt"
                    self.save_plot(image_filename)
                    self.save_current_run(run_filename)

                    f_runs.write('\t'.join(
                        [str(run_id),
                         image_filename[len(out_dir)+1:]] +
                        self.param_str() +
                        [str(epoch),
                         str(curr_step),
                         str(success),
                         str(round(total_time, 3)),
                         str(round(mean_time, 3)),
                         str(round(train_loss, 3)),
                         str(round(test_loss, 3)),
                         str(round(train_tpr, 3)),
                         str(round(train_fpr, 3)),
                         str(round(test_tpr, 3)),
                         str(round(test_fpr, 3)),
                         ]) +
                                 '\n')
                    f_runs.flush()
                    prev_step = curr_step
                    run_id += 1
                    if run_id >= num_runs:
                        break
            f_runs.close()

if __name__ == '__main__':
    run = Run()
    run.execute_runs(run.MODE_PSA_RUNS, 10000, resume=True)
