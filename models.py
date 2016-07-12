from keras.models import *
from keras.layers import *
from keras.optimizers import *

def create_deep_net():
    # create model
    model = Sequential()
    model.add(Dense(102, input_shape=(100, 51), init='normal', activation='relu'))
    model.add(Dense(51, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model

def create_LSTM_model():
    # create model
    model = Sequential()
    model.add(LSTM(200, input_shape=(100, 51), activation='tanh'))
    model.add(Dense(2))

    return model


def train_LSTM(model, X_train, Y_train, nb_epoch, batch_size):

    print "\n\nTraining started:\n\n"

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

    model.fit(X_train,
              Y_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=1)