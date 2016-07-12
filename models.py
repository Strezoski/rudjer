from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping


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
    model.add(LSTM(360, input_shape=(1500, 51), activation='tanh'))
    model.add(Dense(2))

    return model


def train_LSTM(model, X_train, Y_train, nb_epoch, batch_size, early_stop=False):

    print "\n\nTraining started:\n\n"

    rms_opt = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=rms_opt, loss='binary_crossentropy', metrics=["accuracy"])

    if early_stop:

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train,
                  Y_train,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_split=0.2,
                  verbose=1,
                  callbacks=[early_stopping])

    else:
        model.fit(X_train,
                  Y_train,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_split=0.2,
                  verbose=1)