import pickle

import numpy as np
from sklearn import svm

import utils
import vgg16
from constants import DIR_MODEL, DIR_TEST, LABELS


def test():
    X_test, y_test, paths = utils.read_process_images(
        DIR_TEST, LABELS, return_paths=True)
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    X_test_fc6 = vgg16.extract_fc6_features(X_test, verbose=True)
    print(f'X_test_fc6 shape: {X_test_fc6.shape}')

    # load model from disk
    loaded_model = pickle.load(open(DIR_MODEL, 'rb'))

    # predict at test data
    y_pred = loaded_model.predict(X_test_fc6)

    print('Y test = ')
    print(y_test)
    print('Y predict = ')
    print(y_pred)

    scores = loaded_model.score(X_test_fc6, y_test)
    print("accuracy = %f" % scores)

    # Print false predictions
    diff = y_test != y_pred
    if np.any(diff):
        print('false predictions:')
        print(paths[diff])

    # Print confidence score of each sample
    print('confidence scores:')
    print(loaded_model.decision_function(X_test_fc6))


if __name__ == '__main__':
    test()
