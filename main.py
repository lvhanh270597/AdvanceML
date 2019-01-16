import utils
import vgg16
from sklearn import svm
import pickle

labels = ('bike', 'non-bike')
dir_train = './images/train'
dir_test = './images/test'

X_train, y_train = utils.read_process_images(dir_train, labels)
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

X_test, y_test = utils.read_process_images(dir_test, labels)
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

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
print("accuracy = %f" %scores)
''' 
Best Model: LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
'''