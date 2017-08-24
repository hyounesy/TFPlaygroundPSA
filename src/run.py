import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
from matplotlib import colors
from dataset import DataSet
from nn import Classifier


class Run:
    """
    A single run
    """
    num_samples = 200
    noise_range = [0.0, 0.5]
    training_ratio_range = [0.1, 0.9]
    batch_size_range = [1, 30]
    learning_rates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    regularization_rates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    def __init__(self):
        self.dataset_name = DataSet.DATA_CIRCLE
        self.noise = 0.0
        self.training_ratio = 0.5
        self.batch_size = 10

        self.learning_rate = 0.3
        self.num_hidden = 2
        self.num_hidden_neuron = 3
        self.activation_h = Classifier.ACTIVATION_TANH  # activaion function for hidden layers
        self.regularization_type = Classifier.REGULARIZATION_NONE
        self.regularization_rate = 0.0

    def randomize(self):
        pass


if __name__ == '__main__':

    batch_size = 30
    dataset_name = DataSet.DATA_CIRCLE
    num_samples = 200
    noise = 0.0

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
