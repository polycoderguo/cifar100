import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureExtraction:
	def __init__(self, data_input):
		self.x = data_input
	def get_color_pix_r(self):
		per = [len(set([str(x) for x in img.reshape(1024,3)]))/1024 for img in self.x]    	
		return np.array(per)

	def get_color_sat(self):
		sat_array = []
		for array in self.x:
			array = array.reshape(1024,3)
			sat = np.array([max(abs(pix[0]-pix[1]),abs(pix[0]-pix[2]),abs(pix[2]-pix[1])) for pix in array])
			sat_array.append(np.mean(sat))
		return np.array(sat_array)
	def to_grayscale(self):
		gray_img = [cv2.cvtColor(image.reshape(32,32,3), cv2.COLOR_BGR2GRAY).astype(float) for image in self.x]
		gray_img = np.array(gray_img).reshape(self.x.shape[0], 32*32)
		return gray_img

	def get_img_gradient(self):
		'''Default methods are all the three functions in cv2.  '''
		gray = np.array([cv2.cvtColor(image.reshape(32,32,3), cv2.COLOR_BGR2GRAY).astype(float) for image in self.x]).reshape(self.x.shape[0], 32*32)
		grad = np.array([np.concatenate((cv2.Laplacian(img,cv2.CV_64F),cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5),cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5))) for img in gray])
		return grad.reshape(self.x.shape[0],32*32*3)

def to_standardize(x_train, x_test):
	std = StandardScaler()
	x_train_std  = std.fit_transform(x_train)
	x_test_std = std.transform(x_test)
	return x_train_std, x_test_std
	
def to_PCA(x_train, x_test):
	pca = PCA(0.9)
	x_train_pca = pca.fit_transform(x_train)
	x_test_pca = pca.transform(x_test)
	return x_train_pca, x_test_pca
		
