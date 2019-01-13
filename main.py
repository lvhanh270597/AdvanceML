import utils
import vgg16

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
