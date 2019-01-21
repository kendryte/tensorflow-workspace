import tensorflow as tf
import os
import re
import sys
import argparse
import base_func
import importlib
import models
from tensorflow.python.framework import graph_util

def freeze_graph(model_def,input_dir, output_graph):
    sess = tf.InteractiveSession()
    meta_file, ckpt_file = base_func.get_model_filenames(input_dir)

    network = importlib.import_module(model_def)

    images_placeholder = tf.placeholder(tf.float32,shape=(None,224,224,3),name='input')

    logits, _ = network.inference(images_placeholder, keep_probability=0, 
        phase_train=False, class_num=1000)


    ckpt_dir_exp = os.path.expanduser(input_dir)


    meta_file = os.path.join(ckpt_dir_exp, meta_file)
    ckpt_file = os.path.join(ckpt_dir_exp, ckpt_file)

    print("meta-file is %s" % meta_file)


    saver = tf.train.Saver(tf.global_variables())
   

    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图


    output_node_names = "MobileNetV1/Bottleneck2/BatchNorm/Reshape_1"


    with tf.Session() as sess:
   
        saver.restore(sess, ckpt_file) #恢复图并得到数据
        # sess.run(embeddings,feed_dict=feed_dict)


        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']


        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点         # for op in graph.get_operations():        #     print(op.name, op.values())


def main(args):
    freeze_graph(args.model_def,args.ckpt_dir,args.output_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.mobilenet_v1')
    parser.add_argument('ckpt_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
