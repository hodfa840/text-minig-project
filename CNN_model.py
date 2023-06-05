from data_loading import *
from data_preprocessing import *
from tokenizing import *
import os
from tensorflow.keras.optimizers.legacy import Adam

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from data_loading import *
from data_preprocessing import *
from tokenizing import *
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input, Dropout, Dense

import itertools
from sklearn.model_selection import train_test_split

train_data, val_test_data, train_classes, val_test_classes = train_test_split(data_sequences, data_class,test_size = 0.3, stratify=data_class,random_state=42)

test_data, val_data, test_classes, val_classes = train_test_split(val_test_data, val_test_classes, test_size = 0.5, stratify=val_test_classes,random_state=42)

 
inputs = Input(shape=(10,))
embedding_layer = Embedding(input_dim=num_words, output_dim=100, trainable=True, input_length=10, weights=[embedding_wights])(inputs)
#updated in the notebook
cnn_model = Sequential()
# The Embedding layer
cnn_model.add(Embedding(input_dim=num_words, output_dim=100, trainable=True, input_length=10, weights=[embedding_wights]))
cnn_model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
#cnn_model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(rate=0.15))
# The second one dimensional convolutional layer (32,4,same,relu)
cnn_model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
#cnn_model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
# The second Max pooling layer (2)
cnn_model.add(MaxPooling1D(pool_size=2))
# The second Dropout layer (10%)
cnn_model.add(Dropout(rate=0.15))
# The third one dimensional convolutional layer (32,4,same,relu)
cnn_model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
cnn_model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
# The third Max pooling layer (2)
cnn_model.add(MaxPooling1D(pool_size=2))
# The third Dropout layer (10%)
cnn_model.add(Dropout(rate=0.20))
# The Flattening layer
cnn_model.add(Flatten())
# The First Dense Layer (256,relu)
cnn_model.add(Dense(256, activation='relu'))
# The Second Dense Layer or Prediction layer (1,sigmoid)
cnn_model.add(Dense(10, activation='softmax'))
# Compiling the Model using the Binary_Crossontropy as a loss function and accuracy as a meseaure and Adam as an Optimizer
cnn_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
