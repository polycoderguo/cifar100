
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

class Augmentation:
	def __init__ (self, data_input,target_input,augment_size):
		
		self.x = data_input
		self.y = target_input
		self.augment_size = augment_size
	
	def data_augmentation(self):
		train_size = self.x.shape[0]
        	#train_1 = x_1.shape[0]	
		image_generator = ImageDataGenerator(
			rescale = 1./255.,
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			zoom_range = 0.05, 
			width_shift_range=0.07,	
			height_shift_range=0.07,
			horizontal_flip=True,
			vertical_flip=False, 
			data_format="channels_last",
			zca_whitening=True)
        	# fit data for zca whitening
		image_generator.fit(self.x, augment=True)
        	# get transformed images	
		randidx = np.random.randint(train_size, size=self.augment_size)	
		x_augmented = self.x[randidx].copy()
		y_augmented = self.y[randidx].copy()
		x_augmented = image_generator.flow(x_augmented, np.zeros(self.augment_size),
                                    batch_size=self.augment_size, shuffle=False).next()[0]
        	# append augmented data to trainset
		x = np.concatenate((self.x, x_augmented))
		y = np.concatenate((self.y, y_augmented))
        	#train_size = x_0.shape[0]
        	#test_size = x_test.shape[0]
		return x,y



