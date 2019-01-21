from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
from os import environ
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import base_func
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import pickle
from scipy import misc

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def main(args):
    
    image_size = (args.image_size, args.image_size)

    top1 = 0.0
    top5 = 0.0
    
    with tf.Graph().as_default() as graph:      
        with tf.Session() as sess:    
            base_func.load_model(args.model)

            input_image = sess.graph.get_tensor_by_name('input:0')
            output = sess.graph.get_tensor_by_name('MobileNetV1/Bottleneck2/BatchNorm/Reshape_1:0')

            if (os.path.isdir(args.model)):
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
     
            img = np.array(misc.imread(args.image, mode='RGB'))
            # img = base_func.crop(img, False, args.image_size)
            img = misc.imresize(img, image_size, interp='bilinear')
            img = img / 255.
            images = [img]
            
            feed_dict={input_image:images}
            if (os.path.isdir(args.model)):
                feed_dict={input_image:images,phase_train_placeholder:False}

            logits = sess.run(output,feed_dict=feed_dict)
            pred = _softmax(logits[0,:])

            des_idx = np.argsort(pred)

            with open("data/names.list","r") as f:
                lines = f.readlines()
            # with open("data/names2.list","w") as f1:
            #     for k in range(1000):
            #         f1.writelines(lines[k].split(":")[1])

            for j in range(5):
                print("%.2f%%--%s" % (pred[des_idx[999-j]]*100,lines[des_idx[999-j]].strip()))                
       
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('image', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='data/eagle.jpg')
    parser.add_argument('--image_size', type=int,
        help='image size.', default=224)          

    return parser.parse_args(argv)
  

if __name__ == '__main__':
   args = parse_arguments(sys.argv[1:])
   main(args)
