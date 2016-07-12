import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from dataset_manip import *
from models import *

seed = 7
numpy.random.seed(seed)

# X_train, y_train = load_dataset(dataset_path="/home/gjorgji/Desktop/proteinsDataset/sequenceParts/reducedProperties.txt",
#                                 feature_delimiter=';',
#                                 feature_member_delimiter=',',
#                                 features_per_entry=100,
#                                 verbose=True,
#                                 lstm_type=True)

X_train, y_train = load_dataset(dataset_path="/home/gjorgji/Desktop/reducedProperties_padded_test_start.txt",
                                feature_delimiter=';',
                                feature_member_delimiter=',',
                                features_per_entry=1500,
                                verbose=True,
                                lstm_type=True,
                                variable_length=False,
                                max_length=1500)

model = create_LSTM_model()

print model.summary()

train_LSTM(model, X_train, y_train, 20, 68)
