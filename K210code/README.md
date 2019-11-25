### *pb* file to *kmodel* file

Make a directory named `ncc`. Download [nncase](<https://github.com/kendryte/nncase/releases>) tool and uncompress it to `ncc`.

#### *pb* file to tflite

Copy the pretrained model `mobilenetv1_1.0.pb` in `pretrained` directory to `ncc/bin`. 

Enter `ncc/bin` directory.

```shell
toco --graph_def_file=mobilenetv1_1.0.pb --output_file=mobilenetv1_1.0.tflite --output_format=TFLITE --input_shape=1,224,224,3 --input_arrays=inputs --output_arrays=MobileNetV1/Bottleneck2/BatchNorm/Reshape_1 --inference_type=FLOAT
```

#### tflite to kmodel

Enter `ncc` directory and place a few pictures of your dataset into `ncc/dataset` directory.

```shell
./ncc compile ./bin/mobilenetv1_1.0.tflite ./bin/mobilenetv1_1.0.kmodel -i tflite -o kmodel --dataset ./dataset/
```

**Note**: Pictures in `ncc/dataset` are used for quantization. They should cover all classes of your dataset.

### Prepare image for test

Convert an image, for example `eagle.jpg`, to a C file.

```python
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('eagle.jpg')
img = np.transpose(img,[2,0,1]) # KPU requires NCHW format, 
								# while Tensorflow requires NHWC.
with open('image.c','w') as f:
    print('const unsigned char gImage_image[]={', file=f)
    print(', '.join([str(i) for i in img.flatten()]), file=f)
    print('};', file=f)
```

### Test

Copy the `K210code` directory to `kendryte-standalone-sdk/src`. Build and download to KD233 to check the results.

**Note**: `develop` branch of `kendryte-standalone-sdk` is required.

