from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os
import numpy


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


def create_LSTM_model(cur_input_shape=(1500, 51)):
    # create model
    model = Sequential()
    # model.add(Masking(mask_value=0. ,input_shape=(1500, 51)))
    model.add(Masking(mask_value=0., input_shape=cur_input_shape))
    # model.add(LSTM(1500,activation='tanh', inner_activation='hard_sigmoid', consume_less='mem'))
    model.add(LSTM(400, activation='tanh', inner_activation='hard_sigmoid', consume_less='mem', return_sequences=True))
    model.add(LSTM(200, activation='tanh', inner_activation='hard_sigmoid', consume_less='mem'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))

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


def train_LSTM(model, x_train, y_train, x_validation, y_validation, nb_epoch, batch_size, learning_rate=0.0001,
               early_stop=False, train_batch=False):

    print "\n\nTraining started:\n\n"

    rms_opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=rms_opt, loss='binary_crossentropy', metrics=["accuracy"],)

    if early_stop:  # model.fit is always used with early stop parameter

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        
        model.fit(x_train,
                  y_train,
                  validation_data=(x_validation, y_validation),
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  callbacks=[early_stopping])

    else:

        if train_batch:
            train_batch_by_batch(model, x_train, y_train, x_validation, y_validation, nb_epoch, batch_size)
        else:       
            model.fit(x_train,
                      y_train,
                      validation_data=(x_validation, y_validation),
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      verbose=1)


def train_batch_by_batch(model, x_train, y_train, x_validation, y_validation, nb_epoch, batch_size):
     """
     Train one by one batch of samples.
     This is for memory optimization: only one batch is transferred to GPU instead of whole dataset.
     """
     indexes = range(len(x_train))

     for epoch in range(nb_epoch):
            print ('Epoch {}/{}'.format(epoch+1, nb_epoch))

            np.random.shuffle(indexes)
            batches = split_in_batches(len(x_train), batch_size)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = indexes[batch_start:batch_end]

                x_batch = [x_train[b] for b in batch_ids]
                x_batch = numpy.array(x_batch)

                y_batch = [y_train[b] for b in batch_ids]
                y_batch = numpy.array(y_batch)

                loss, acc = model.train_on_batch(x_batch, y_batch)
                if batch_index == (len(batches)-1):  # last batch: print accuracy
                    print 'train_loss: %.3f - train_acc: %3f' % (float(loss), float(acc))

            loss, acc = model.test_on_batch(x_validation, y_validation)
            print 'val_loss: %.3f - val_acc: %3f' % (float(loss), float(acc))


def split_in_batches(size, batch_size):
    """
    :param size:
    :param batch_size:
    :return: a list of batch indices (tuples of indices)
    """
    nb_batch = int(np.ceil(size / float(batch_size)))

    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def train_LSTM_kfold(model, x, y, learning_rate=0.0001, nb_epoch=40, batch_size=64, n_folds=3, train_batch=False):

    print "\n\n[7] ["+str(n_folds)+"-FOLD] Training started:\n"

    rms_opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=rms_opt, loss='binary_crossentropy', metrics=["accuracy"],)

    kf = KFold(len(y), n_folds=n_folds)

    current_validation_iteration = 1

    for train, test in kf:

        print "[TRAINING] ["+str(n_folds)+"-FOLD] Currently in #"+str(current_validation_iteration)+" fold.\n"

        current_validation_iteration += 1

        x_train = x[train]
        x_validation = x[test]
        y_train = y[train]
        y_validation = y[test]
        
        if train_batch:
            train_batch_by_batch(model, x_train, y_train, x_validation, y_validation, nb_epoch, batch_size)
        else:  # fit model and score
            model.fit(x_train,
                      y_train,
                      validation_data=(x_validation, y_validation),
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      verbose=1)