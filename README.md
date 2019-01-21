Tensorflow Workspace for K210
======

## Classifier for ImageNet 
1. 下载ImageNet数据集，按照说明解压缩训练数据集到文件夹ILSVRC2012\_img\_train，内含1000个子文件夹，每个子文件夹的命名为其分类代号（类似n02484975），每个子文件夹内为该分类的训练数据  
2. mobilenet v1定义文件：mobilenetv1/models/mobilenet\_v1.py，需要注意由于K210不支持tensorflow的SAME padding，所以在stride=2时先固定padding一圈0，然后再进行stride=2的卷积（padding=VALID）  
3. 训练脚本 mobilenetv1/run\_mobilenet\_v1.sh，根据需要修改其中的参数，然后运行  
4. freeze\_graph.py将训练ckpt转成pb文件，命令格式如下：  
   python mobilenetv1/freeze\_graph.py model.mobilenet\_v1 ckpt\_fold pb\_file  
5. 测试在ImageNet验证集上的性能，下载验证集，将文件按类别解压好（与训练集类似），运行 python mobilenetv1/validation\_imagenet.py pb\_file（or ckpt folder） val\_set\_fold  
6. 预测单张图片，python mobilenetv1/predict\_one\_pic.py pb\_file（or ckpt folder） pic
