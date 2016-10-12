from __future__ import division
from dataset_manip import *
from models import *

print "\n[DEBUG] Checking input arguments...\n"

if len(sys.argv) != 10:
    print "[ERROR] Input arguments invalid! \n"
    print "Usage: python proteins_exp.py validation_split random_seed epochs learning_rate batch_size k_fold train_batch dataset_path test_run\n"
    print "Example: \n python proteins_exp.py 0 6 3 0.0001 12 2 0 /home/gjorgji/Desktop/reducedProperties_padded_test_hlf.txt 100"

else:

    validation_split = float(sys.argv[1])
    seed = int(sys.argv[2])
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    k_fold = int(sys.argv[6])
    train_batch = bool(int(sys.argv[7]))
    dataset_path = str(sys.argv[8])

    if int(str(sys.argv[9])) != 0:
        test_run = True
    else:
        test_run = False

    print "Dataset: " + str(dataset_path)
    print "Validation split: " + str(validation_split)
    print "Random seed #: " + str(seed)
    print "Epochs: " + str(epochs)
    print "Learning rate:  " + str(learning_rate)
    print "Batch size: " + str(batch_size)
    print "K-Fold coeff: " + str(k_fold)
    print "Train batch by batch: " + str(train_batch)
    print "Test run: " + str(test_run)

    numpy.random.seed(seed)

    dataset_path = dataset_path
    feature_delimiter = ';'
    feature_member_delimiter = ','
    validation_split = validation_split

    if k_fold <= 1:
        x_train, x_validation, y_train, y_validation = load_dataset(dataset_path=dataset_path,
                                                                    feature_delimiter=feature_delimiter,
                                                                    feature_member_delimiter=feature_member_delimiter,
                                                                    seed=seed,
                                                                    kfold=False,
                                                                    validation_split=validation_split,
                                                                    verbose=True,
                                                                    lstm_type=True,
                                                                    test_run=test_run)

        f = open('rfData_train', 'w')
        for i in range(x_train.shape[0]):
            count = np.zeros(4)
            for j in range(x_train.shape[1]):
                x_element = x_train[i][j]
                if np.sum(x_element) == 1:
                    idx = np.where(x_element == 1)[0][0]
                    count[idx] += 1
            for j in range(4):
                f.write(str(count[j] / np.sum(count)) + ",")
            f.write(str(int(y_train[i][0])) + "\n")
        f.close()

        f = open('rfData_valid', 'w')
        for i in range(x_validation.shape[0]):
            count = np.zeros(4)
            for j in range(x_validation.shape[1]):
                x_element = x_validation[i][j]
                if np.sum(x_element) == 1:
                    idx = np.where(x_element == 1)[0][0]
                    count[idx] += 1
            for j in range(4):
                f.write(str(count[j] / np.sum(count)) + ",")
            f.write(str(int(y_validation[i][0])) + "\n")
        f.close()

        model = create_LSTM_model(cur_input_shape=(x_train.shape[1], x_train.shape[2]))

        print "\n\n[5] Model summary: \n"
        print model.summary()

        print "\n\n[6] Training started: \n"
        train_LSTM(model, x_train, y_train, x_validation, y_validation, epochs, batch_size,
                   learning_rate, train_batch=train_batch)

    else:

        x_train, y_train = load_dataset(dataset_path=dataset_path,
                                        feature_delimiter=feature_delimiter,
                                        feature_member_delimiter=feature_member_delimiter,
                                        seed=seed,
                                        kfold=True,
                                        verbose=True,
                                        lstm_type=True,
                                        test_run=test_run)

        model = create_LSTM_model()

        print "\n\n[5] Model summary: \n"
        print model.summary()


        print "\n\n[6] ["+str(k_fold)+"-FOLD] Model created and training started: \n"

        train_LSTM_kfold(model=model,
                         x=x_train,
                         y=y_train,
                         nb_epoch=epochs,
                         batch_size=batch_size,
                         n_folds=k_fold,
                         train_batch=train_batch)
