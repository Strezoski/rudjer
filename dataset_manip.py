import numpy
from datetime import datetime
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


def convert_feature_members_float(feature):
    """
    Converts the string values read from the file to floats.
    :param feature: The feature vector for the encoded character
    :return: tmp_array: The converted array.
    """
    tmp_array = []
    for string in feature:
        tmp_array.append(float(string))

    return tmp_array


def load_dataset(dataset_path, feature_delimiter, feature_member_delimiter, seed, kfold=False, validation_split=0.2, verbose=True, lstm_type=True,
                 test_run=False):
    """
    Generates the datasets into numpy arrays compatible with Keras.
    :param dataset_path: The path to the textual file containing the dataset.
    :param feature_delimiter: The feature vectors delimiter (number of chars in sequence).
    :param feature_member_delimiter: The delimiter inside the feature arrays.
    :param seed: Random seed number for getting consistent results.
    :param verbose: Print out shapes and duration.
    :return: (X_train, Y_train) tuple: Keras compatible dataset.
    """

    t = datetime.now()
    print "\n[1] Dataset loading started... \n"
    print "Time: " + str(t)

    y_train_aux = []
    X_train = numpy.zeros(shape=(1, 1, 1))
    counter = 0

    limit = 100

    for line in open(dataset_path):
        if counter == 0:

            speciments_number = int(line.split(feature_member_delimiter)[0])
            max_dimension = int(line.split(feature_member_delimiter)[1])
            feature_dims = int(line.split(feature_member_delimiter)[2].strip())

            if test_run:
                X_train = numpy.zeros(shape=(limit, max_dimension, feature_dims))
            else:
                X_train = numpy.zeros(shape=(speciments_number, max_dimension, feature_dims))

            counter = counter + 1

            print "\n[2] Dataset info:"
            print "Number of speciments:" + str(speciments_number)
            print "Max dimension: " + str(max_dimension)
            print "Feature dims: " + str(feature_dims)

        else:

            if test_run:
                if counter >= limit:
                    y_train_aux.append(int(line.split(feature_delimiter)[-1]))
                    break

            line_helper = []
            split_line = line.split(feature_delimiter)
            protein_class = int(split_line[-1])
            protein_features = split_line[1:-1]
            nonzero_position = int(split_line[0])

            zero_array = numpy.zeros(shape=(nonzero_position, feature_dims))

            for feature in protein_features:
                temp_array = convert_feature_members_float(feature.split(feature_member_delimiter))
                line_helper.append(temp_array)

            line_helper = numpy.array(line_helper)

            X_train[counter - 1] = numpy.concatenate((zero_array, line_helper), axis=0)

            y_train_aux.append(protein_class)

            counter += 1
            # Diagnostic prints
            # print len(protein_features)
            # print len(y_train)
            # print X_train[counter-1]
            # print X_train[counter-1][nonzero_position+1]
            # print X_train[counter-1][nonzero_position-1]
            # print X_train[counter-1].shape
            # print X_train[counter-1][nonzero_position]
            # print X_train[counter-1][nonzero_position-1]
            # print X_train[counter-1][nonzero_position+1]
            # print line_helper.shape
            # print X_train.shape

    y_train = numpy.array(y_train_aux)

    print "\n[3] Dataset loaded...\n"

    if lstm_type:
        Y_train = np_utils.to_categorical(y_train)
    else:
        Y_train = y_train

    if not kfold:

        print "\n[DEBUG] Non K-Fold dataset prepared \n"
        x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=validation_split, random_state=seed)

        if verbose:
            print "\n[4] Dataset summary:"
            print "Training data shape: " + str(x_train.shape)
            print "Training labels data shape:" + str(y_train.shape)
            print "Validation data shape: " + str(x_validation.shape)
            print "Validation labels data shape:" + str(y_validation.shape)
            print "Time for conversion: " + str(datetime.now() - t) + "\n\n"

        return x_train, x_validation, y_train, y_validation

    else:
        print "\n[DEBUG] K-Fold dataset prepared \n"

        x_train = X_train
        y_train = Y_train

        if verbose:
            print "\n[4] Dataset summary:"
            print "Training data shape: " + str(x_train.shape)
            print "Training labels data shape:" + str(y_train.shape)
            print "Time for conversion: " + str(datetime.now() - t) + "\n\n"

        return x_train, y_train