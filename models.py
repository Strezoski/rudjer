from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from keras.models import *
from keras.layers import *
from keras.optimizers import *
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

def train_LSTM(model, X_train, Y_train, x_validation, y_validation, nb_epoch, batch_size, learning_rate=0.0001, early_stop=False,):

    print "\n\nTraining started:\n\n"

    rms_opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08)
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

def train_LSTM_kfold(random_seed, X_train, Y_train, create_function=create_sigmoid_LSTM_model, nb_epoch=40, batch_size=64, n_folds=3):

    estimator = KerasClassifier(build_fn=create_function, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    kfold = StratifiedKFold(y=Y_train, n_folds=n_folds, shuffle=True, random_state=random_seed)
    results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))