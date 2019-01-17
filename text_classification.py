import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
# print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # num_words keeps the 10,000 most frequent words while discarding the rest, for memory purposes
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0]) # here's what the first review looks like

## Convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1 
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
														value=word_index["<PAD>"],
														padding='post',
														maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
														value=word_index["<PAD>"],
														padding='post',
														maxlen=256)

# print(len(train_data[0]), len(train_data[1]))  # 256, 256
# print(train_data[0]) 

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

# create a validation set
# When training, we want to check the accuracy of the model on data it hasn't seen before. Create a validation set by setting apart 10,000 examples from the original training data. 
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model for 40 epochs in mini-batches of 512 samples
history = model.fit(partial_x_train,
					partial_y_train,
					epochs=40,
					batch_size=512,
					validation_data=(x_val, y_val),
					verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)  # this fairly naive approach achieves an accuracy of about 87%. With more advanced approaches, it can get up to 95%


# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()
dict_keys = (['acc', 'val_acc', 'loss', 'val_loss'])

# import matplotlib.pyplot as plt  # it crashes here due to "ImportError: No module named '_tkinter', please intall the python3-tk package". This is probably machine specific

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# # "bo" is for blue dot
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # "b" is for solid blue line
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.clf() # clear figure
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()