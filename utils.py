import os

import numpy as np
from PIL import Image


def read_process_images(dir, labels):
    X = []
    y = []

    for i, label in enumerate(labels):
        for fn in os.listdir(os.path.join(dir, label)):
            img = Image.open(os.path.join(dir, label, fn))

            # Ensure 224x224x3
            assert len(img.getbands()) == 3
            img = img.resize((224, 224))

            X.append(np.array(img))
            y.append(i)

    return np.array(X), np.array(y)
