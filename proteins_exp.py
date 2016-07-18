import numpy
import sys
from dataset_manip import *
from models import *

print  "\n[DEBUG] Checking input arguments...\n"

if len(sys.argv) != 9:
    print "[ERROR] Input arguments invalid! \n"
    print "Usage: python proteins_exp.py validation_split random_seed epochs learning_rate batch_size k_fold dataset_path test_run\n"
else:

    validation_split = float(sys.argv[1])
    seed = int(sys.argv[2])
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    k_fold = int(sys.argv[6])
    dataset_path = str(sys.argv[7])
    test_run = bool(sys.argv[8])

    print "Dataset: " + str(sys.argv[7])
    print "Validation split: " + str(validation_split)
    print "Radnom seed #: " + str(seed)
    print "Epochs: " + str(epochs)
    print "Learning rate:  " + str(learning_rate)
    print "Batch size: " + str(batch_size)
    print "K-Fold coeff: " + str(k_fold)
    print "Test run: " + str(test_run)


    numpy.random.seed(seed)

    dataset_path = dataset_path
    feature_delimiter = ';'
    feature_member_delimiter = ','
    validation_split = validation_split

    x_train, x_validation, y_train, y_validation = load_dataset(dataset_path=dataset_path,
                                                                feature_delimiter=feature_delimiter,
                                                                feature_member_delimiter=feature_member_delimiter,
                                                                seed=seed,
                                                                validation_split=validation_split,
                                                                verbose=True,
                                                                lstm_type=True,
                                                                test_run=test_run)

    model = create_sigmoid_LSTM_model()

    print "\n\n [5] Model summary: \n"
    print model.summary()

    print "\n\n [6] Training started: \n"
    train_LSTM(model, x_train, y_train, x_validation, y_validation, epochs, batch_size, learning_rate)
