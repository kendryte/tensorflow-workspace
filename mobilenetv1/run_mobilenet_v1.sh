#!/bin/bash

python3 train_softmax.py \
--model_def models.mobilenet_v1 \
--data_dir /data/datasets/ImageNet2012/ILSVRC2012_img_train/ \
--pretrained_model "../pretrained/mobilenetv1_1.0.pb"\
--gpu_memory_fraction 0.85 \
--gpus 1 \
--image_size 224 \
--logs_base_dir backup_classifier \
--models_base_dir backup_classifier \
--batch_size 100 \
--epoch_size 5000 \
--learning_rate -1 \
--max_nrof_epochs 50 \
--class_num 1000 \
--use_fixed_image_standardization \
--optimizer MOM \
--learning_rate_schedule_file data/learning_rate.txt \
--keep_probability 1.0 
