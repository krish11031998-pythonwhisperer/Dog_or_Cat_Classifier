import tensorflow as tf
import pandas
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def image_classifier():
	image_classifier = tf.keras.Sequential()
	image_classifier.add(tf.keras.layers.Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
	image_classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
	image_classifier.add(tf.keras.layers.Flatten())
	image_classifier.add(tf.keras.layers.Dense(128,activation = 'relu'))
	image_classifier.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
	image_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
	return image_classifier


def image_generator(dir,classifier):
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)
	test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	train_set = train_datagen.flow_from_directory(
		dir+'/dataset/training_set',
		target_size=(64, 64),
		batch_size=32,
		class_mode='binary')

	test_set = test_datagen.flow_from_directory(
		dir+'/dataset/test_set',
		target_size=(64, 64),
		batch_size=32,
		class_mode='binary')

	classifier.fit_generator(
		train_set,
		steps_per_epoch=8000,
		epochs=10 ,
		validation_data=test_set,
		validation_steps=2000)

	return classifier
def save_classifier(classifier:object):
	saving_classifier = classifier.to_json()
	with open('image_classifier.json','w') as classifier_to_json:
		classifier_to_json.write(saving_classifier)
	classifier.save_weights('image_classifier.h5')
	print("The classifier was successfully saved")


def load_classifier(name_json,name_weights):
	with open(name_json,'r') as json_classifier:
		loaded_classifier = json_classifier.read()
	classifier = tf.keras.models.model_from_json(loaded_classifier)
	classifier.load_weights(name_weights)
	classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
	print("successfully loaded the classifier")
	return classifier


def predict(filename,classifier):
	test_image = image.load_img(filename,target_size=(64,64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis = 0)
	prediction = classifier.predict(test_image)
	# train_set.class_indices
	if prediction[0][0] == 1:
		result = 'dog'
	else:
		result = 'cat'

	return result 


if __name__ == '__main__':
	path = '/Users/krishnavenkatramani/Desktop/Deep_Learning/CNN/Dog_or_Cat _Classifier'
	name_json = 'image_classifier.json'
	name_weights = 'image_classifier.h5'
	if os.path.isfile(name_json) and os.path.isfile(name_weights):
		classifier = load_classifier(path+'/'+name_json,path+'/'+name_weights)
	else:
		classifier = image_classifier()
		image_generator(path,classifier)
		save_classifier(classifier)
	#finally 
	single_prediction_filedir = 'dataset/single_prediction'
	for filename in os.listdir(single_prediction_filedir):
		print("The predicted animal in the given image {} is {}".format(filename,predict(single_prediction_filedir+'/'+filename,classifier)))


