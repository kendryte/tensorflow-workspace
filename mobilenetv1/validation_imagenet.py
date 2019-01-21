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
    environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    image_size = (args.image_size, args.image_size)

    dataset = base_func.get_dataset(args.data_dir)

    val_set = dataset
    image_list, label_list = base_func.get_image_paths_and_labels(val_set)
        
    nrof_classes = len(val_set)
    val_image_num = len(image_list)

    top1 = 0.0
    top5 = 0.0
    
    with tf.Graph().as_default() as graph:      
        with tf.Session() as sess:    
            base_func.load_model(args.model)

            input_image = sess.graph.get_tensor_by_name('input:0')
            output = sess.graph.get_tensor_by_name('MobileNetV1/Bottleneck2/BatchNorm/Reshape_1:0')

            if (os.path.isdir(args.model)):
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            for i in range(val_image_num):    
                print(image_list[i])            
                # images = base_func.load_data([image_list[i]], False, False, args.image_size)
                img = np.array(misc.imread(image_list[i], mode='RGB'))
                # img = base_func.crop(img, False, args.image_size)
                img = misc.imresize(img, image_size, interp='bilinear')
                img = img / 255.
                images = [img]

                feed_dict={input_image:images}
                if (os.path.isdir(args.model)):
                    feed_dict={input_image:images,phase_train_placeholder:False}

                logits = sess.run(output,feed_dict=feed_dict)
                pred = _softmax(logits[0,:])
                # print(logits)
                des_idx = np.argsort(pred)
                # des_data = np.sort(logits)
                # print(des_data[0,995:])

                if (des_idx[nrof_classes-1]) == label_list[i]:
                    top1 += 1
                for j in range(5):
                    if (des_idx[nrof_classes-1-j]) == label_list[i]:
                        top5 += 1 
                        break    
                print("%05d th pic have been validated, top1 = %.2f%%  top5 = %.2f%% " % (i+1,top1/(i+1)*100.,top5/(i+1)*100.))              
                 

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str,
        help='Model definition. ckpt folder or pb file', default='mobilenet_v1.pb')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/iamgenet_val')
    parser.add_argument('--image_size', type=int,
        help='image size.', default=224)          
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--gpus', type=str,
        help='Indicate the GPUs to be used.', default='3')  


    return parser.parse_args(argv)
  

if __name__ == '__main__':
   args = parse_arguments(sys.argv[1:])
   print('gpu device ID: %s'%args.gpus)
   main(args)
