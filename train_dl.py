import numpy as np

from  data_process import PreprocessDataset
from  data_aug import Augmentation
from data_eng import FeatureExtraction
import data_eng
from resnet import Resnet

label_0 = ['bridge', 'castle', 'house', 'road', 'skyscraper']
label_1 = ['cloud', 'forest', 'mountain', 'plain','sea']


target_0 = ['bridge', 'castle'] 
target_1 = ['cloud', 'forest']
train_0 = ['house', 'road', 'skyscraper']
train_1 = [ 'mountain', 'plain','sea']

x_train, x_test, y_train, y_test = PreprocessDataset(target_0, target_1, train_0, train_1, True)
print('initial shape')
print(x_train.shape, x_test.shape)

aug_train = Augmentation(x_train, y_train, 14400)
x_train, y_train = aug_train.data_augmentation()
print('after augmentation shape')
print(x_train.shape, x_test.shape)

#feature_train = FeatureExtraction(x_train)
#feature_test = FeatureExtraction(x_test)
#gray = feature_train.to_grayscale()
#sat = feature_train.get_color_sat().reshape(-1,1)
#pix = feature_train.get_color_pix_r().reshape(-1,1)
#grad = feature_train.get_img_gradient()
#print(gray.shape, sat.shape, pix.shape, grad.shape)
#x_train = np.concatenate((feature_train.to_grayscale(), feature_train.get_color_sat().reshape(-1,1), feature_train.get_color_pix_r().reshape(-1,1), feature_train.get_img_gradient()),axis = 1)
#x_test = np.concatenate((feature_test.to_grayscale(), feature_test.get_color_sat().reshape(-1,1), feature_test.get_color_pix_r().reshape(-1,1), feature_test.get_img_gradient()),axis = 1)
#print(x_train.shape)
#print(np.concatenate((gray, sat, pix), axis =1).shape)
#x_train = np.concatenate((gray, grad, sat, pix), axis =1)
#print('after data engineering x_train shape, x_test shape')
#print(x_train.shape, x_test.shape)

#x_train, x_test = data_eng.to_standardize(x_train, x_test)
#print('std done')

#x_train,x_test = data_eng.to_PCA(x_train, x_test)
#print('pca done')

res = Resnet(x_train, x_test, y_train, y_test)
accurate = res.train()

print(accurate)

