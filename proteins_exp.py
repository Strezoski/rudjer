import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from dataset_manip import *
from models import *

X_train, y_train = load_dataset(dataset_path="/home/gjorgji/Desktop/proteinsDataset/sequenceParts/reducedProperties.txt",
                                feature_delimiter=';',
                                feature_member_delimiter=',',
                                features_per_entry=100,
                                verbose=True)
print y_train

pass

model = create_LSTM_model()

print model.summary()

train_deep_net(model, X_train, y_train, 10, 12)
