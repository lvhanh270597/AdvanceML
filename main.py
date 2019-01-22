import utils
import vgg16
from sklearn import svm
import pickle, os

labels = ('bike', 'non-bike')
dir_train = './images/train'
dir_test = './images/test'
dir_model = 'model.sav'

def train_section():
    if (dir_model not in os.listdir()):
        X_train, y_train = utils.read_process_images(dir_train, labels)
        print(f'X_train shape: {X_train.shape}')
        print(f'y_train shape: {y_train.shape}')
        X_train_fc6 = vgg16.extract_fc6_features(X_train, verbose=True)
        print(f'X_train_fc6 shape: {X_train_fc6.shape}')
        # declare model
        clf = svm.LinearSVC()
        # training
        clf.fit(X_train_fc6, y_train)
        # write down model at disk
        pickle.dump(clf, open(dir_model, 'wb'))

def test_section():
    X_test, y_test = utils.read_process_images(dir_test, labels)
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    X_test_fc6 = vgg16.extract_fc6_features(X_test, verbose=True)
    print(f'X_test_fc6 shape: {X_test_fc6.shape}')
    # load model from disk
    loaded_model = pickle.load(open(dir_model, 'rb'))
    print('Y test = ')
    print(y_test)
    print('Y predict = ')
    # predict at test data
    print(loaded_model.predict(X_test_fc6))
    scores = loaded_model.score(X_test_fc6, y_test)
    print("accuracy = %f" %scores)

def main():
    train_section()
    test_section()

main()