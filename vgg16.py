from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

model = VGG16(include_top=True, weights='imagenet')
model_fc6 = Model(inputs=model.input,
                  outputs=model.get_layer('fc1').output)


def extract_fc6_features(images, verbose=False):
    global model, model_fc6

    X = preprocess_input(images, mode='caffe')

    if verbose:
        return model_fc6.predict(X, batch_size=1, verbose=1)
    else:
        return model_fc6.predict(X)
