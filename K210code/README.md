### *pb* file to C file

Download [nncase](https://github.com/kendryte/nncase) tool and uncompress it. You'll get `ncc` directory.

#### *pb* file to tflite

Copy the pretrained model `mobilenetv1_1.0.pb` in `pretrained` directory to `ncc\bin`. Rename it to `mobilenetv1.pb` to avoid the function name error in the following steps.

Enter `ncc\bin` directory from *cmd*.

```shell
 .\toco.exe --input_file=mobilenetv1.pb --input_format=TENSORFLOW_GRAPHDEF --output_file=mobilenetv1.tflite --output_format=TFLITE --input_shape=1,224,224,3 --input_array=inputs --output_array=MobileNetV1/Bottleneck2/BatchNorm/Reshape_1 --inference=FLOAT
```

#### tflite to C code

Enter `ncc` directory and place the dataset into `ncc\dataset` directory.

```shell
.\ncc.exe -i tflite -o k210code --dataset .\dataset\ .\bin\mobilenetv1.tflite .\bin\mobilenetv1.c
```

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

Copy the `K210code` directory to `kendryte-standalone-sdk\src`. Build and download to KD233 to check the results.

**Note**: `develop` branch of `kendryte-standalone-sdk` is required.