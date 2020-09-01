import os
import sys
import numpy as np

from skimage.io import imread

sys.path.insert(0, '.')
if os.environ.get('TF_KERAS'):
    import efficientnet.tfkeras as efn
    from tensorflow.keras.models import load_model
else:
    import efficientnet.keras as efn
    from keras.models import load_model
    

if __name__ == "__main__":
    model = efn.EfficientNetB0()