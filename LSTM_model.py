from data_loading import *
from data_preprocessing import *
from tokenizing import *
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects
import itertools
from sklearn.model_selection import train_test_split

train_data, val_test_data, train_classes, val_test_classes = train_test_split(data_sequences, data_class,test_size = 0.3, stratify=data_class,random_state=42)

test_data, val_data, test_classes, val_classes = train_test_split(val_test_data, val_test_classes, test_size = 0.5, stratify=val_test_classes,random_state=42)

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=num_words, output_dim=100, trainable=True, input_length=10, weights=[embedding_wights]))
model_lstm.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.2), 'concat'))
model_lstm.add(Dropout(0.4))
model_lstm.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))
model_lstm.add(Dropout(0.4))
model_lstm.add(LSTM(64, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(64, activation='sigmoid'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(10, activation='softmax'))
model_lstm.compile(loss ='categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])
stopping = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience = 1, verbose = 1, factor = 0.1, min_delta=0.001, min_lr = 0.00001)



