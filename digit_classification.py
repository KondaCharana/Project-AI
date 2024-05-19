"""Digit classification.ipynb


Original file is located at
    https://colab.research.google.com/drive/1dkWsrS5QkYFz3dfvnZ4nW8zNIgIEpQZf
"""

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train.shape

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
fig, axs = plt.subplots(4, 4, figsize = (10, 10))
plt.gray()
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number      {}'.format(y_train[i]))
fig.show()

print(x_train[3])

print(x_train)

print(x_train.ndim)

print(y_train)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #1 represents the color channel(gray here)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) #it becomes 4 dimensional
input_shape = (28, 28, 1)

#print(x_train)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 #everything will be b/w 0 and 1
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape)) #img we choosen is 2d so conv2d #input shape=28,28,1
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #covert whole matrix into single line-1D
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax)) #tells the probability to our given number

model.compile(optimizer='adam',                     #adding optimizer
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=1) #epochs is no.of iterations

model.evaluate(x_test, y_test)