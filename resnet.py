from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

class Resnet:
	def __init__(self, x_train, x_test, y_train, y_test):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
	def model(self):

		base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32,32,3))
		resnet_input_train = preprocess_input(self.x_train)
		resnet_input_test = preprocess_input(self.x_test)
		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- here we have 2 classes
		predictions = Dense(2, activation='softmax')(x)

		# this is the model we will train
		model = Model(inputs=base_model.input, outputs=predictions)
		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional DenseNet201 layers
		for layer in base_model.layers:
			layer.trainable = False

		# compile the model (should be done *after* setting layers to non-trainable)
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
		# train the model on the new data for a few epochs
		model.fit(resnet_input_train, to_categorical(self.y_train, num_classes=2), batch_size=64, epochs=2,
				validation_data=(resnet_input_test, to_categorical(self.y_test, num_classes=2)),
				verbose=2, shuffle=True)
		# at this point, the top layers are well trained and we can start fine-tuning
		# convolutional layers from Resnet. We will freeze the bottom N layers
		# and train the remaining top layers.
		for layer in model.layers[:0]:
			layer.trainable = False
		for layer in model.layers[0:]:
			layer.trainable = True
		model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
		model.fit(resnet_input_train, to_categorical(self.y_train, num_classes=2), batch_size=64, epochs=150,
				validation_data=(resnet_input_test, to_categorical(self.y_test, num_classes=2)),verbose=2, shuffle=True)

		y_tr_predict = model.predict(resnet_input_train)
		y_te_predict = model.predict(resnet_input_test)
		accuracy = accuracy_score(self.y_test, y_te_predict)
		return accuracy
