import numpy
from datetime import datetime
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

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

def load_dataset(dataset_path, feature_delimiter, feature_member_delimiter, features_per_entry, verbose=False, lstm_type=True, variable_length=False, max_length=100):
    """
    Generates the datasets into numpy arrays compatible with Keras.
    :param dataset_path: The path to the textual file containing the dataset.
    :param feature_delimiter: The feature vectors delimiter (number of chars in sequence).
    :param feature_member_delimiter: The delimiter inside the feature arrays.
    :param features_per_entry: Number of features per character in the sequence.
    :param verbose: Print out shapes and duration.
    :param variable_length: If True than we set a zero padding on every sequence.
    :param max_length: Specifies the max length og the sequence we apply zero padding to.
    :return: (X_train, Y_train) tuple: Keras compatible dataset.
    """

    t = datetime.now()
    print "\n\nDataset loading started... \n"

    X_train_aux = []
    y_train_aux = []

    for line in open(dataset_path):

        line_helper = []

        split_line = line.split(feature_delimiter)
        protein_class = int(split_line[features_per_entry])

        protein_features = split_line[0:features_per_entry]
        print len(protein_features)
        for feature in protein_features:
            temp_array = convert_feature_members_float(feature.split(feature_member_delimiter))
            # print len(temp_array)
            # print temp_array
            # if sum(temp_array)==0:
            #     temp_array = convert_feature_members_float(feature.split(feature_member_delimiter)[0:51])


            line_helper.append(temp_array)

        # print "Line helper len:"+str(len(line_helper))

        X_train_aux.append(line_helper)
        y_train_aux.append(protein_class)
        # print len(protein_features)
        # print len(y_train)

    X_train = numpy.array(X_train_aux)
    y_train = numpy.array(y_train_aux)

    if lstm_type:
        Y_train = np_utils.to_categorical(y_train)
    else:
        Y_train = y_train

    if variable_length:
        X_train = pad_sequences(X_train, maxlen=max_length)

    if verbose:
        print "Training data shape: "+str(X_train.shape)
        print "Labels data shape:" +str(y_train.shape)
        print "Time for conversion: "+str(datetime.now() - t)+"\n\n"

    return X_train, Y_train