#!/usr/bin/env python
import tensorflow as tf
import sys
import os


print(sys.version)
print(os.getenv("VIRTUAL_ENV"))


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    pass
