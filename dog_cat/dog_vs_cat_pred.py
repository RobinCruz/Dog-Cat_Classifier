#D/B Dogs and Cats
import tensorflow as tf
#Importing Model and layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml

#Loading YAML to Create model

yaml_file = open("DogVsCat.yaml","r")
Classifier = yaml_file.read()
yaml_file.close()
#Adding Model Attributs
Classifier = model_from_yaml(Classifier)
#Adding Weights
Classifier.load_weights("DogVsCat.h5")
#Classifier is Loaded

#To Get Summary of the network
Classifier.summary()
from keras.utils.vis_utils import plot_model
plot_model(Classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from ann_visualizer.visualize import ann_viz

ann_viz(Classifier,view=False,filename="dogvscat.gv",title="DOG VS CAT")

#Load Test Image
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('testimg.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
#Prediction
result = Classifier.predict(test_image)

if result[0][0] >= 0.5:
	pred = 'dog'
else:
	pred = 'cat'
print(pred)
