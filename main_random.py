import utils
import vgg16
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle

labels = ('bike', 'non-bike')
dir = './image2'

X, y = utils.read_process_images(dir, labels)

for cnt in range(1):
    print("lan lap thu %d: " %(cnt+1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, stratify=y)

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')

    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print(y_test)

    X_train_fc6 = vgg16.extract_fc6_features(X_train, verbose=True)
    print(f'X_train_fc6 shape: {X_train_fc6.shape}')

    X_test_fc6 = vgg16.extract_fc6_features(X_test, verbose=True)
    print(f'X_test_fc6 shape: {X_test_fc6.shape}')

    # declare model
    clf = svm.LinearSVC()
    # training
    clf.fit(X_train_fc6, y_train)
    # write down model at disk
    filename = 'model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # load model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    print('Y test = ')
    print(y_test)
    print('Y predict = ')
    # predict at test data
    print(loaded_model.predict(X_test_fc6))
    scores = loaded_model.score(X_test_fc6, y_test)
    print("accuracy = %.2f%%" %(scores*100))
