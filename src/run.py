import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
from matplotlib import colors
from dataset import DataSet
from nn import Classifier
import random

class Run:
    """
    A single run
    """
    num_samples = 200
    range_noise = [0.0, 0.5]
    range_training_ratio = [0.1, 0.9]
    range_batch_size = [1, 30]
    learning_rates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    regularization_rates = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    range_hidden = [0, 6]
    range_hidden_neuron = [1, 8]

    def __init__(self):
        # dataset parameters
        self.dataset_name = DataSet.DATA_CIRCLE
        self.noise = 0.0
        self.training_ratio = 0.5
        # classifier parameters
        self.batch_size = 10
        self.learning_rate = 0.3
        self.num_hidden = 2
        self.num_hidden_neuron = 3
        self.activation_h = Classifier.ACTIVATION_TANH  # activaion function for hidden layers
        self.regularization_type = Classifier.REGULARIZATION_NONE
        self.regularization_rate = 0.0
        self.data = None
        self.classifier = None

    def randomize(self):
        # dataset parameters
        self.dataset_name = random.choice(DataSet.data_names)
        self.noise = random.uniform(*self.range_noise)
        self.training_ratio = random.uniform(*self.range_training_ratio)
        # classifier parameters
        self.batch_size = random.randint(*self.range_batch_size)
        self.learning_rate = random.choice(self.learning_rates)
        self.num_hidden = random.randint(*self.range_hidden)
        self.num_hidden_neuron = random.randint(*self.range_hidden_neuron) if self.num_hidden > 0 else 0
        self.activation_h = random.choice(Classifier.activations_names)
        self.regularization_type = random.choice(Classifier.regularization_names)
        self.regularization_rate = random.choice(self.regularization_rates)

    def one_run(self):
        self.data = DataSet(self.dataset_name, self.num_samples, self.noise)
        self.data.training_ratio = self.training_ratio
        self.classifier = Classifier()
        self.classifier.batch_size = self.batch_size
        self.classifier.learning_rate = self.learning_rate
        self.classifier.num_hidden = self.num_hidden
        self.classifier.num_hidden_neuron = self.num_hidden_neuron
        self.classifier.activation_h = self.activation_h
        self.classifier.regularization_type = self.regularization_type
        self.classifier.regularization_rate = self.regularization_rate
        self.classifier.build()

        stats = self.classifier.train(self.data, max_steps = 4001, stat_steps=[100, 500, 1000, 2000, 4000])
        # print(stats)
        return stats

    def param_str(self):
        return "\t".join([self.dataset_name,
                          '{:0.2f}'.format(self.noise),
                          str(self.batch_size),
                          str(self.learning_rate),
                          str(self.num_hidden),
                          str(self.num_hidden_neuron),
                          self.activation_h,
                          self.regularization_type,
                          str(self.regularization_rate)
                          ])

    def param_names(self):
        return ['dataset','noise', 'batch_size', 'learning_rate', 'hidden_layers', 'neurons', 'activation', 'regularization', 'regularization_rate']

    def save_plot(self, filename):
        # matplotlib.interactive(False)
        # plot the resulting classifier
        colormap = colors.ListedColormap(["#f59322", "#e8eaeb", "#0877bd"])
        x_min, x_max = -6, 6  # data.points[:, 0].min() - 1, data.points[:, 0].max() + 1
        y_min, y_max = -6, 6  # data.points[:, 1].min() - 1, data.points[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        z = self.classifier.predict_y(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig = plt.figure(figsize=(4, 4), dpi=75)
        # plt.imshow(z, cmap=colormap, interpolation='nearest')
        plt.contourf(xx, yy, z, cmap=colormap, alpha=0.8)
        point_color = self.data.labels
        plt.scatter(self.data.points[:, 0], self.data.points[:, 1], c=point_color, s=30, cmap=colormap)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        fig.savefig(filename)

if __name__ == '__main__':
    run = Run()
    with open('../output/runs.txt', 'w') as f:
        f.write('index' + '\t' +
                '\t'.join(run.param_names()) + '\t' +
                '\t'.join(['step', 'train_loss', 'test_loss']) +
                '\n')
        index = 0
        for i in range(20):
            run.randomize()
            stats = run.one_run()
            for stat in stats:
                f.write(str(index) + '\t' +
                        run.param_str() + '\t' +
                        '\t'.join([str(stat['step']), str(stat['train_loss']), str(stat['test_loss'])]) +
                        '\n')
                index += 1
            # print(run.param_str())
            run.save_plot('../output/images/' + str(index - 1)+".png")


