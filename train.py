import pickle

from sklearn import svm

import utils
import vgg16
from constants import DIR_MODEL, DIR_TRAIN, LABELS


def train():
    X_train, y_train = utils.read_process_images(DIR_TRAIN, LABELS)
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')

    X_train_fc6 = vgg16.extract_fc6_features(X_train, verbose=True)
    print(f'X_train_fc6 shape: {X_train_fc6.shape}')

    # declare model
    clf = svm.LinearSVC()
    # training
    clf.fit(X_train_fc6, y_train)
    # write down model at disk
    pickle.dump(clf, open(DIR_MODEL, 'wb'))


if __name__ == '__main__':
    train()
