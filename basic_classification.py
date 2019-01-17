import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt 

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
     
print(train_images.shape)
print(len(train_labels))

## ! none of the plots will show up while matplotlib.use("agg") is there
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)

# scale the values of pixels to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_images[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])

## Build the model
# Setup the layers
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), # 1st layer: tranforms the images from 2d 28x28 array into 1d array of 28*28=784 pixels
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax),
	])

## Compile the model
# loss function, optimizer, metrics
model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

## Train the model
# start training
model.fit(train_images, train_labels, epochs=5)

## Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

## Make predictions
predictions = model.predict(test_images)
print(predictions[0])
# a prediction is an array of 10 numbers. These describe the 'confidence' of the model that the image corresopnds to each of the 10 different articles of clothing. We can see which label has the highest confidence value with:
# np.argmax(predictions[0])
# That'll return 9. So the model is most confident that this image is an ankle boot, or class_names[9], and we can check the label to see this is correct with: test_labels[0]

## Graph it to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
		100*np.max(predictions_array),
		class_names[true_label]),
	color=color)


def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

