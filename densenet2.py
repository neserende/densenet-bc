# -*- coding: utf-8 -*-
"""
In this code, we implement DenseNet as described in Huang et al's paper
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Conv2D, AveragePooling2D, GlobalAveragePooling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

#loading CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#normalizing pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

#converting labels to one-hot encodings using keras method to_categorical
#in this dataset, we have 10 classes so we put 10 as second parameter
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#separating training set into train and validation
#we use random_state = some integer to get the same split every run
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

#k, growth rate is also one of the hyperparameters
k = 12 #growth rate
input_shape = (32, 32, 3) #shape of images in CIFAR-10 dataset
num_layers = 40 #denoted as L in paper
compression = 0.5
dropout_rate = 0.2 #in paper, if no data augmentation, they add dropout after each conv except first

def h_l(x, k, channel_num, dropout_rate):
  #1x1 conv for bottleneck
  x = BatchNormalization()(x)
  x = ReLU()(x)
  #we use 4 as bottleneck width
  x = Conv2D(k * 4, (1, 1), kernel_regularizer = l2(1e-4))(x)
  #add dropout
  x = Dropout(dropout_rate)(x)

  #3x3 conv
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(channel_num, (3, 3), padding='same')(x)
  #add dropout
  x = Dropout(dropout_rate)(x)

  return x

#x: non linear transformation of concatenation of feature maps produced in previous layers
    #H_l[x0, x1, ..., x(l-1)]
#k: growth rate
def dense_block(x, block_layer_num, k, channel_num, dropout_rate):
  #for each layer do the operations
  for i in range(block_layer_num):
    #pass x through composite function h
    x_l = h_l(x, k, channel_num, dropout_rate)
    #concatenate x with current output
    x = concatenate([x, x_l])
    #update number of channels
    channel_num += k

  #return new value of x and channel_num
  return x, channel_num

def transition_layer(x, channel_num, compression, dropout_rate):
  #update the number of channels with compression constant given
  channel_num = int(channel_num * compression)

  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(channel_num, (1, 1), padding='same', kernel_regularizer = l2(1e-4))(x) #1x1 conv
  #add dropout
  x = Dropout(dropout_rate)(x)
  x = AveragePooling2D((2, 2), strides=(2, 2))(x) #2x2 avg pooling

  return x, channel_num

def create_densenet_model(input_shape, k, compression, dropout_rate):
  #we are implementing DenseNet with BC
  #3 dense blocks with equal number of layers = 6 layers each
  block_layer_num = 6 #this value changes when num_layers is changed
  channel_num = k

  inputs = Input(input_shape)
  #before entering dense blocks, convolution with (2 * growth rate) output channels
  x = Conv2D(2 * k, (3, 3), padding = 'same', kernel_regularizer = l2(1e-4))(inputs)

  #dense blocks
  #first dense block and transition layer
  x, channel_num = dense_block(x, block_layer_num, k, channel_num, dropout_rate)
  x, channel_num = transition_layer(x, channel_num, compression, dropout_rate)

  #second dense block and transition layer
  x, channel_num = dense_block(x, block_layer_num, k, channel_num, dropout_rate)
  x, channel_num = transition_layer(x, channel_num, compression, dropout_rate)

  #third dense block and global avg pooling
  x, channel_num = dense_block(x, block_layer_num, k, channel_num, dropout_rate)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = GlobalAveragePooling2D()(x)

  #softmax to predict outputs
  #first parameter is number of classes
  outputs = Dense(10, activation = 'softmax')(x)

  return Model(inputs, outputs)

#create the model and store it in a variable
model = create_densenet_model(input_shape, k, compression, dropout_rate)

#display the structure of the model
model.summary()

#compile the model
model.compile(optimizer = Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy', tf.keras.metrics.F1Score(name='f1_score')])

#start training
train_history = model.fit(x_train, y_train, epochs=40, batch_size=64, validation_data=(x_val, y_val))

epochs = range(1, len(train_history.history['accuracy']) + 1)

#plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, train_history.history['val_accuracy'], label='Validation Accuracy')
#plt.plot(epochs, y = [train_history.history['f1_score']], label='Training F-Score')
plt.legend()

plt.title('Training Metrics Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')

#plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_history.history['loss'], label='Training Loss')
plt.plot(epochs, train_history.history['val_loss'], label='Validation Loss')
#plt.plot(epochs, y = [train_history.history['f1_score']], label='Training F-Score')
plt.legend()

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#test the model
test_history = model.evaluate(x_test, y_test, verbose = 1)
