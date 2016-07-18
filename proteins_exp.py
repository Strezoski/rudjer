import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from dataset_manip import *
from models import *

seed = 7
numpy.random.seed(seed)

dataset_path = "/home/gjorgji/Desktop/reducedProperties_padded_test_hlf.txt"
feature_delimiter = ';'
feature_member_delimiter = ','

x_train, x_validation, y_train, y_validation = load_dataset(dataset_path=dataset_path,
                                                            feature_delimiter=feature_delimiter,
                                                            feature_member_delimiter=feature_member_delimiter,
                                                            seed=seed,
                                                            verbose=True,
                                                            lstm_type=True,
                                                            test_run=True)

model = create_sigmoid_LSTM_model()

print model.summary()

train_LSTM(model, x_train, y_train, x_validation, y_validation, 20, 16)
