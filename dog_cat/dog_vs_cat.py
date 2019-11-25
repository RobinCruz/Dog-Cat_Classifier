#D/B Dogs and Cats
import tensorflow as tf
#Importing Model and layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml

#Creating Convolution Network
Classifier = Sequential()
#Layering
Classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu")) #(Output dim,kernel size,stride)
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Flatten())

#Adding to Artificial Neural Network with 2 layers
Classifier.add(Dense(activation='relu',units=128))
Classifier.add(Dense(activation='sigmoid',units=1))#Sigmoid is used as activation for getting probability since we need probability of the image being with a dog or a cat

#Compiling the Whole Network
Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Data PreProcessing
#Data Augmentation is used to increse the amount of different data to train and test by altering slightly here Horizontally Flipping
from keras.preprocessing.image import ImageDataGenerator
#Generating Object for trainingset
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#Generating Object for testingset
test_datagen = ImageDataGenerator(rescale=1./255)
#Generating
train_set = train_datagen.flow_from_directory('training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='binary')

#Training the Network
#from IPython.display import display
#from PIL import Image
Classifier.fit_generator(train_set,steps_per_epoch=8000,epochs=10,validation_data=test_set,validation_steps=800)


#Saving Model as YAML
yaml_model = Classifier.to_yaml()
with open("DogVsCat.yaml","w") as yaml_file:
	yaml_file.write(yaml_model)
#Saving weights to HDF5
Classifier.save_weights("DogVsCat.h5")

#Testing
'''
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('testimg.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = Classifier.predict(test_image)
train_set.class_indices
if result[0][0] >= 0.5:
	pred = 'dog'
else:
	pred = 'cat'
print(pred)
'''
#To Visualise
from ann_visualizer.visualize import ann_viz

ann_viz(Classifier,view=True,filename="dogvscat.gv",title="DOG VS CAT")

#Loading YAML to Create model

yaml_file = open("DogVsCat.yaml","r")
Classifier = yaml_file.read()
yaml_file.close()
#Adding Model Attributs
Classifier = model_from_yaml(Classifier)
#Adding Weights
Classifier.load_weights("DogVsCat.h5")
#Then use
			
