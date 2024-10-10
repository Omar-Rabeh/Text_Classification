import time
import numpy as np
import tensorflow as tf

t0 = tf.constant(4) # scalar

t1 = tf.constant([2.0, 3.0, 4.0]) # 1D tensor

t2 = tf.constant([ [1,2],[3,4],[5,6] ],dtype=tf.float16) # 1D tensor
print(t2[2,0])
print(t2[:,0].numpy())