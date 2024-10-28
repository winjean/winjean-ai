import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if len(tf.config.experimental.list_physical_devices('GPU')):
    print("has GPU")

elif len(tf.config.experimental.list_physical_devices('CPU')):
    print("has CPU")
else:
    print("NONE")