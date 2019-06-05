# cifar100
An interesting project is presenting here:

Purpose: binary image classification

Data: cifar100 dataset

Dataset contains 20 superclasses and each has 5 classes in it. 
An example is given as follow:

Superclass   			Classes <br>
aquatic mammals			beaver, dolphin, otter, seal, whale <br>
fish				aquarium fish, flatfish, ray, shark, trout <br>

large man-made outdoor things	bridge, castle, house, road, skyscraper
large natural outdoor scenes	cloud, forest, mountain, plain, sea

More information can be found in https://www.cs.toronto.edu/~kriz/cifar.html.

Challenges here:
As shown in the example above, large man-made outdoor things and large natual outdoor scenes are distinct enough but also share some simialrity. 
Is it possible to let model learns from only part of the classes and correctly classify the remaining targets?
E.g.: by only feeding with [bridge, castle, house] and [cloud, forest, mountain] as training data, can the model successully differenciate [road, skyscraper] with [plain, sea]?

Questions to be answered:
1) with partial information, will deep learning always outperform the traditional machine learning?
2) by doing image feature engineering and augmentation, can the performance of machine learning performance be improved?

Results:


