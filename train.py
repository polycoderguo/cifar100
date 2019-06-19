import numpy as np

from  data_process import PreprocessDataset
from  data_aug import Augmentation
from data_eng import FeatureExtraction
import data_eng

#import numpy as np
from numpy import genfromtxt
from  models import modeltrainer

label_0 = ['bridge', 'castle', 'house', 'road', 'skyscraper']
label_1 = ['cloud', 'forest', 'mountain', 'plain','sea']


target_0 = ['bridge', 'castle'] 
target_1 = ['cloud', 'forest']
train_0 = ['house', 'road', 'skyscraper']
train_1 = [ 'mountain', 'plain','sea']


target_0 = []
target_1 = []
train_0 = []
train_1 = []

for i, first in enumerate(label_0):
	for j ,second in enumerate(label_0[i+1:]):
		target_0.append([first, second])
		train_0.append(list(set(label_0)-set([first,second])))
for i, first in enumerate(label_1):
        for j ,second in enumerate(label_1[i+1:]):
                target_1.append([first, second])
                train_1.append(list(set(label_1)-set([first,second])))

result = []

for i in range(10):
	for j in range(10):
		print(target_0[i], target_1[j], train_0[i], train_1[j])
		x_train, x_test, y_train, y_test = PreprocessDataset(target_0[i], target_1[j], train_0[i], train_1[j], False)
		
		print('initial shape')
		print(x_train.shape, x_test.shape)
		aug_train = Augmentation(x_train, y_train, 14400)
		x_train, y_train = aug_train.data_augmentation()
		print('after augmentation shape')
		print(x_train.shape, x_test.shape)

		feature_train = FeatureExtraction(x_train)
		feature_test = FeatureExtraction(x_test)
		#gray = feature_train.to_grayscale()
		#sat = feature_train.get_color_sat().reshape(-1,1)
		#pix = feature_train.get_color_pix_r().reshape(-1,1)
		#grad = feature_train.get_img_gradient()
		#print(gray.shape, sat.shape, pix.shape, grad.shape)
		x_train = np.concatenate((feature_train.to_grayscale(), feature_train.get_color_sat().reshape(-1,1), feature_train.get_color_pix_r().reshape(-1,1), feature_train.get_img_gradient()),axis = 1)
		x_test = np.concatenate((feature_test.to_grayscale(), feature_test.get_color_sat().reshape(-1,1), feature_test.get_color_pix_r().reshape(-1,1), feature_test.get_img_gradient()),axis = 1)
#print(x_train.shape)
#print(np.concatenate((gray, sat, pix), axis =1).shape)
#x_train = np.concatenate((gray, grad, sat, pix), axis =1)
		print('after data engineering x_train shape, x_test shape')
		print(x_train.shape, x_test.shape)

		x_train, x_test = data_eng.to_standardize(x_train, x_test)
		print('std done')

		x_train,x_test = data_eng.to_PCA(x_train, x_test)
		print('pca done')


		#x_train  = genfromtxt('train_ml.csv', delimiter = ',')
		#x_test = genfromtxt('test_ml.csv', delimiter = ',')
		#y_train = genfromtxt('train_y.csv', delimiter = ',')
		#y_test = genfromtxt('test_y.csv', delimiter = ',')
		clf = modeltrainer(x_train, x_test, y_train, y_test)
		accuracy_score = clf.train_GradientBoost()
		print(target_0[i], target_1[j], train_0[i], train_1[j])
		print(accuracy_score)
		result.append(accuracy_score)
result = np.array(result)
np.savetxt("result_gdb_ml.csv", result, delimiter=",")

#np.savetxt("train_ml.csv", x_train, delimiter=",")
#np.savetxt("test_ml.csv", x_test,delimiter="," )
#np.savetxt("train_y.csv", y_train, delimiter=",")
#np.savetxt("test_y.csv", y_test,delimiter="," )
#print('done data prepaation')
		
