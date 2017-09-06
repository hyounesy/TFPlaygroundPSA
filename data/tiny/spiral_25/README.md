## SPIRAL_25 dataset
This is a tiny data set of the first 50 records (10 random hyper-parameter combinations at 5 epochs) from the [complete dataset of 10,000 rows](https://drive.google.com/uc?id=0Bz2L2qpV9PICRWtBYWY1VkFuZWs&export=download)) and for demonstration purpose only.

| file | description |
|----|----|
| input.txt | input data generated with spiral shape at noise 25%. seven features (X<sub>1</sub>, X<sub>2</sub>, X<sub>1</sub><sup>2</sup>, X<sub>2</sub><sup>2</sup>, X<sub>1</sub>X<sub>2</sub>, sin(X<sub>1</sub>), sin(X<sub>2</sub>)) and labels (ground truth) for 200 input points. The ratio of the training to test is a varying hyper-parameter |
| index.txt | values for the hyper-parameters used for each run and summary statistics. There are 5 rows per parameter combination correponding to 5 different epochs (25, 50, 100, 200, 400) |
| paramInfo.txt | information about each column of index.txt (hyper-parameters + output statistics)|
| runs/[i].txt | one file for each run: predicted labels the points at the i'th run |
| images/[i].png | one plot for each run output: predicted classification. training and test points are represented with white and black stroke respectively.|

The following images are the classification output plot for random hyper-parameters at different epochs.

| epoch: 25 | epoch: 50 | epoch: 100 | epoch: 200 | epoch: 400 |
|----|----|----|----|----|
![](images/0.png)|![](images/1.png)|![](images/2.png)|![](images/3.png)|![](images/4.png)|
![](images/5.png)|![](images/6.png)|![](images/7.png)|![](images/8.png)|![](images/9.png)|
![](images/10.png)|![](images/11.png)|![](images/12.png)|![](images/13.png)|![](images/14.png)|
![](images/15.png)|![](images/16.png)|![](images/17.png)|![](images/18.png)|![](images/19.png)|
![](images/20.png)|![](images/21.png)|![](images/22.png)|![](images/23.png)|![](images/24.png)|
![](images/25.png)|![](images/26.png)|![](images/27.png)|![](images/28.png)|![](images/29.png)|
![](images/30.png)|![](images/31.png)|![](images/32.png)|![](images/33.png)|![](images/34.png)|
![](images/35.png)|![](images/36.png)|![](images/37.png)|![](images/38.png)|![](images/39.png)|
![](images/40.png)|![](images/41.png)|![](images/42.png)|![](images/43.png)|![](images/44.png)|
![](images/45.png)|![](images/46.png)|![](images/47.png)|![](images/48.png)|![](images/49.png)|
