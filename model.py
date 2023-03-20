from data_loading import *
from data_preprocessing import *
from tokenizing import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers.legacy import Adam

import itertools
from sklearn.model_selection import train_test_split

train_data, val_test_data, train_classes, val_test_classes = train_test_split(data_sequences, data_class,test_size = 0.3, stratify=data_class,random_state=42)

test_data, val_data, test_classes, val_classes = train_test_split(val_test_data, val_test_classes, test_size = 0.5, stratify=val_test_classes,random_state=42)

activation=tf.nn.swish


get_custom_objects().update({'swish': Activation(swish)})
# Define SiLU activation function
get_custom_objects().update({'swish': Activation(swish)})

# Build the model
inputs = Input(shape=(10,))
x = Embedding(input_dim=num_words, output_dim=100, trainable=True, input_length=10, weights=[embedding_wights])(inputs)
x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), merge_mode='concat')(x)
x = Dropout(0.4)(x)
x = LSTM(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(x)
x = Dropout(0.4)(x)
x = LSTM(64, return_sequences=False, recurrent_dropout=0.2, dropout=0.2)(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='swish')(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



