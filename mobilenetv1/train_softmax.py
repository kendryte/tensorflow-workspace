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

def main(args):
    environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    base_func.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = base_func.get_dataset(args.data_dir)

    train_set, val_set = dataset, []
        
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = base_func.get_image_paths_and_labels(train_set)

        assert len(image_list)>0, 'The training set should not be empty'
        
        val_image_list, val_label_list = base_func.get_image_paths_and_labels(val_set)

        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
        
        nrof_preprocess_threads = 4
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        image_batch, label_batch = base_func.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))
        
        print('Building training graph')
        
        # Build the inference graph
        logits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, class_num=args.class_num, 
            weight_decay=args.weight_decay)
  
        prelogits = logits 

        print('class_num=%d' % len(train_set))

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        total_loss = tf.add_n([cross_entropy_mean],name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = base_func.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # var_list += bn_moving_vars
        var_list = list(set(var_list+bn_moving_vars))

        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

        if pretrained_model:
            saver_restore = tf.train.Saver(var_list=var_list)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver_restore.restore(sess, tf.train.latest_checkpoint(pretrained_model))

            # Training and validation loop
            print('Running training')
            nrof_steps = args.max_nrof_epochs*args.epoch_size

            for epoch in range(1,args.max_nrof_epochs+1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step, 
                    total_loss, train_op, args.learning_rate_schedule_file,
                    cross_entropy_mean, accuracy, learning_rate,
                    prelogits, args.random_rotate, args.random_crop, args.random_flip, args.use_fixed_image_standardization)
                # stat['time_train'][epoch-1] = time.time() - t
                
                if not cont:
                    break

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

    return model_dir

  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step, 
      loss, train_op, learning_rate_schedule_file, 
      cross_entropy_mean, accuracy, 
      learning_rate, prelogits,  random_rotate, random_crop, random_flip, use_fixed_image_standardization):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = base_func.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
        
    if lr<=0:
        return False 

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = base_func.RANDOM_ROTATE * random_rotate + base_func.RANDOM_CROP * random_crop + base_func.RANDOM_FLIP * random_flip + base_func.FIXED_STANDARDIZATION * use_fixed_image_standardization
    print('use_fixed_image_standardization=%d' % use_fixed_image_standardization)
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        tensor_list = [loss, train_op, step, prelogits, cross_entropy_mean, learning_rate, accuracy]

        loss_, _, step_, prelogits_, cross_entropy_mean_, lr_, accuracy_  = sess.run(tensor_list, feed_dict=feed_dict)
        
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f\tLr %2.5f' %
              (epoch, batch_number+1, args.epoch_size, duration, loss_, cross_entropy_mean_,  accuracy_, lr_ ))
        batch_number += 1
        train_time += duration

    return True   

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/base_func')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/base_func')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--gpus', type=str,
        help='Indicate the GPUs to be used.', default='2')

    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
        
    parser.add_argument('--class_num_changed', type=bool, default=False,
        help='indicate if the class_num is different from pretrained.')       
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=20)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=100)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=5000)
    parser.add_argument('--class_num', type=int,
        help='Dimensionality of the embedding.', default=1000)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')


    return parser.parse_args(argv)
  

if __name__ == '__main__':
   args = parse_arguments(sys.argv[1:])
   print('gpu device ID: %s'%args.gpus)
   main(args)
