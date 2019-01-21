from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
  
def inference(images, keep_probability, phase_train=True, 
              class_num=1000, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d,slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return mobilenet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, class_num=class_num, reuse=reuse)



def block_14x14(net, outputs, scope=None, reuse=None):
    with tf.variable_scope(scope, "block-14x14", reuse=reuse):
        net = slim.separable_conv2d(net, None, [3, 3],
                                depth_multiplier=1,
                                stride=1,
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                scope='dw_Conv2d')               
        net = slim.conv2d(net, outputs, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,scope='conv')
    return net

def mobilenet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.5,
                        class_num=1000,
                        reuse=None,
                        scope='MobileNetV1'):
    end_points = {}
    net = None
    _l = 0
    with tf.variable_scope(scope, 'MobileNetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d],
                                stride=1, padding='SAME',normalizer_fn=slim.batch_norm): 
                inputs = tf.space_to_batch(inputs,[[1,1],[1,1]],block_size=1,name=None)                                               
                # ------------------------x224------------------------- #
                net = slim.conv2d(inputs, 32, 3, stride=2,padding='VALID',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_3x3'%(_l))
                _l += 1
                # ------------------------x112------------------------- #
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                
                _l += 1
                net = slim.conv2d(net, 64, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = tf.space_to_batch(net,[[1,1],[1,1]],block_size=1,name=None)  
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=2,
                                                padding='VALID',
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                # ------------------------x56-------------------------- #
                net = slim.conv2d(net, 128, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                net = slim.conv2d(net, 128, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = tf.space_to_batch(net,[[1,1],[1,1]],block_size=1,name=None)
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=2,
                                                padding='VALID',
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                # ------------------------x28-------------------------- #
                net = slim.conv2d(net, 256, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                net = slim.conv2d(net, 256, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = tf.space_to_batch(net,[[1,1],[1,1]],block_size=1,name=None)
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=2,
                                                padding='VALID',
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                # ------------------------x14-------------------------- #
                net = slim.conv2d(net, 512, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1

                with tf.variable_scope(scope,'block_repeat_%i'%(_l)):
                    for _k in range(5):
                        net = block_14x14(net,512)
                _l += 1

                net = tf.space_to_batch(net,[[1,1],[1,1]],block_size=1,name=None)
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=2,
                                                padding='VALID',
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))       
                _l += 1
                # -------------------------x7-------------------------- #
                net = slim.conv2d(net, 1024, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                net = slim.separable_conv2d(net, None, [3, 3],
                                                depth_multiplier=1,
                                                stride=1,
                                                padding='SAME',
                                                normalizer_fn=slim.batch_norm,
                                                scope='dw_Conv2d_%i_3x3'%(_l))                       
                _l += 1
                net = slim.conv2d(net, 1024, 1, stride=1, padding='SAME',normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_%i_1x1'%(_l))
                _l += 1
                # ---------------------softmax out---------------------- #
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_%i'%(_l))
                _l += 1                      
            
                net = slim.flatten(net)
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                   scope='Dropout_1')

                net = slim.fully_connected(net, class_num, activation_fn=None,
                                           scope='Bottleneck2', reuse=False)

    return net, None
