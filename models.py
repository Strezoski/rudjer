from keras.models import *
from keras.layers import *
from keras.optimizers import *

def create_LSTM_model():
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(100, 51)))
    model.add(Dropout(0.3))
    model.add(Dense(2))

    # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(optimizer="rmsprop", loss='mse', metrics=["mse", "accuracy"])
    return model


def train_deep_net(model, X_train, Y_train, nb_epoch, batch_size):

    print "\n\nTraining started:\n\n"

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    model.fit(X_train,
              Y_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=1)