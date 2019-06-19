import numpy as np

from  data_process import PreprocessDataset
from  data_aug import Augmentation
from data_eng import FeatureExtraction
import data_eng

label_0 = ['bridge', 'castle', 'house', 'road', 'skyscraper']
label_1 = ['cloud', 'forest', 'mountain', 'plain','sea']


target_0 = ['bridge', 'castle'] 
target_1 = ['cloud', 'forest']
train_0 = ['house', 'road', 'skyscraper']
train_1 = [ 'mountain', 'plain','sea']

x_train, x_test, y_train, y_test = PreprocessDataset(target_0, target_1, train_0, train_1, False)

np.save('raw_train', x_train)

#np.savetxt("raw_train.csv", x_train, delimiter=",")
