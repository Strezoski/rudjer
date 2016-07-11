import numpy
from datetime import datetime
from keras.utils import np_utils

def convert_feature_members_float(feature):
    """
    Converts the string values read from the file to floats.
    :param feature: The feature vector for the encoded character
    :return:
    """
    tmp_array = []
    for string in feature:
        tmp_array.append(float(string))

    return tmp_array

def load_dataset(dataset_path, feature_delimiter, feature_member_delimiter, features_per_entry, verbose=False):
    """
    Generates the datasets into numpy arrays compatible with Keras.
    :param dataset_path: The path to the textual file containing the dataset.
    :param feature_delimiter: The feature vectors delimiter (number of chars in sequence).
    :param feature_member_delimiter: The delimiter inside the feature arrays.
    :param features_per_entry: Number of features per character in the sequence.
    :param verbose: Print out shapes and duration.
    :return:
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

        for feature in protein_features:
            temp_array = convert_feature_members_float(feature.split(feature_member_delimiter))
            # print len(temp_array)
            # print temp_array
            line_helper.append(temp_array)
        # print "Line helper len:"+str(len(line_helper))

        X_train_aux.append(line_helper)
        y_train_aux.append(protein_class)
        # print len(protein_features)
        # print len(y_train)

    X_train = numpy.array(X_train_aux)
    y_train = numpy.array(y_train_aux)
    Y_train = np_utils.to_categorical(y_train)

    if verbose:
        print "Training data shape: "+str(X_train.shape)
        print "Labels data shape:" +str(y_train.shape)
        print "Time for conversion: "+str(datetime.now() - t)+"\n\n"

    return X_train, Y_train
