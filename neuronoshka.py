import tensorflow as tf
import keras as tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
model = Sequential()
model.add(Conv2D(64, 7, input_shape=[512, 512, 1], activation='relu', padding='same'))
model.add(Conv2D(64, 7, activation='relu', padding='same'))
model.add(Conv2D(64, 7, activation='relu', padding='same'))
model.add(Conv2D(64, 7, activation='relu', padding='same'))
model.add(Conv2D(64, 7, activation='relu', padding='same'))
model.add(Conv2D(64, 7, activation='relu', padding='same'))
model.add(Conv2D(1, 1, activation='sigmoid'))
model.compile(optimizer=Adam(), loss='mse')


