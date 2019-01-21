Tensorflow Workspace for K210
======

## Classifier for ImageNet 
1. Download ImageNet dataset, extract it as the instructions to ILSVRC2012\_img\_train, it has 1000 folders, each folder's name represents its class id.
2. The model definition file is `mobilenetv1/models/mobilenet_v1.py`, **ATTENTION**, K210 does not support the method of *SAME PADDING* in tensorflow, so we need to add padding around the image manually before a `stride=2` conv (in this situation, the padding method of conv layer with `stride=2` should be set to *VALID*
3. Modify `mobilenetv1/run_mobilenet_v1.sh` and start your training.
4. Using `freeze_graph.py` to freeze your model from `ckpt` to `pb`, just run `python mobilenetv1/freeze_graph.py model.mobilenet_v1 ckpt_fold pb_file`
5. Test on ImageNet, you need a val dataset of ImageNet, then run `python mobilenetv1/validation_imagenet.py pb_file val_set_fold`
6. Estimate one image, run `python mobilenetv1/predict_one_pic.py pb_file pic`


## ImageNet 分类器示例
1. 下载ImageNet数据集，按照说明解压缩训练数据集到文件夹ILSVRC2012\_img\_train，内含1000个子文件夹，每个子文件夹的命名为其分类代号（类似n02484975），每个子文件夹内为该分类的训练数据  
2. mobilenet v1定义文件：mobilenetv1/models/mobilenet\_v1.py，需要注意由于K210不支持tensorflow的SAME padding，所以在stride=2时先固定padding一圈0，然后再进行stride=2的卷积（padding=VALID）  
3. 训练脚本 mobilenetv1/run\_mobilenet\_v1.sh，根据需要修改其中的参数，然后运行  
4. freeze\_graph.py将训练ckpt转成pb文件，命令格式如下：  
   python mobilenetv1/freeze\_graph.py model.mobilenet\_v1 ckpt\_fold pb\_file  
5. 测试在ImageNet验证集上的性能，下载验证集，将文件按类别解压好（与训练集类似），运行 python mobilenetv1/validation\_imagenet.py pb\_file（or ckpt folder） val\_set\_fold  
6. 预测单张图片，python mobilenetv1/predict\_one\_pic.py pb\_file（or ckpt folder） pic
