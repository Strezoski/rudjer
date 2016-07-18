from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping
import os

def create_deep_net():
    # create model
    model = Sequential()
    model.add(Dense(102, input_shape=(100, 51), init='normal', activation='relu'))
    model.add(Dense(51, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model

def create_tanh_LSTM_model():
    # create model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(1500, 51)))
    model.add(LSTM(850, activation='tanh', consume_less="mem"))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model

def create_tanh_LSTM_model():
    # create model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(1500, 51)))
    model.add(LSTM(850, activation='tanh', consume_less="mem"))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model

def create_tanh_dropout_LSTM_model():
    # create model
    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(1500, 51)))
    model.add(LSTM(850, activation='tanh', inner_activation='tanh', consume_less="mem"))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    return model

def create_sigmoid_LSTM_model():
    # create model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(1500, 51)))
    model.add(LSTM(300, activation='tanh', consume_less="mem"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    return model

def save_model(model, file_name):

    if not os.path.exists('./models/'):
        os.makedirs('./models/')

    model_json = model.to_json()
    open('./models/'+file_name+'.json', 'w').write(model_json)
    model.save_weights('./models/'+file_name+'_weights.h5', overwrite=True)


def train_LSTM(model, X_train, Y_train, x_validation, y_validation, nb_epoch, batch_size, early_stop=False):

    print "\n\nTraining started:\n\n"

    rms_opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=rms_opt, loss='binary_crossentropy', metrics=["accuracy"],)

    if early_stop:

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train,
                  Y_train,
                  validation_data=(x_validation, y_validation),
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  callbacks=[early_stopping])

    else:
        model.fit(X_train,
                  Y_train,
                  validation_data=(x_validation, y_validation),
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1)