import pickle

from sklearn import svm

import utils
import vgg16
from constants import DIR_MODEL, DIR_TEST, LABELS


def test():
    X_test, y_test = utils.read_process_images(DIR_TEST, LABELS)
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    X_test_fc6 = vgg16.extract_fc6_features(X_test, verbose=True)
    print(f'X_test_fc6 shape: {X_test_fc6.shape}')

    # load model from disk
    loaded_model = pickle.load(open(DIR_MODEL, 'rb'))

    print('Y test = ')
    print(y_test)
    print('Y predict = ')
    # predict at test data
    print(loaded_model.predict(X_test_fc6))

    scores = loaded_model.score(X_test_fc6, y_test)
    print("accuracy = %f" % scores)


if __name__ == '__main__':
    test()
