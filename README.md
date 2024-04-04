## GTRSB Classification: PyTorch implementation
The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. The properties of GTSRB include:
- Single-image, multi-class classification problem
- 43 classes
- More than 50,000 images in total

##  Content Structure
* [Description](#description)
* [Installation](#installation)
* [Dataset](#dataset)
* [Train](#train)
* [Evaluation](#evaluation)
* [Deployment](#deployment)
* [References](#references)
* [Licence](#licence)
# Description
I designed my model by adding two Convolutional Neural Network (CNN) layers, effectively deepening the network architecture. This enhancement enabled me to achieve an impressive accuracy of 94.0%. This demonstrates the effectiveness of employing deeper architectures in improving model performance. The trained weight file has been converted into an ONNX file format to facilitate execution within a C language environment. The ONNXRUNTIME C API is employed for running the model efficiently on CPU memory.

# Installation
Download the project:
```git clone https://github.com/jabborov/german_traffic_sign_classification.git```

Install requrement libraries:
```pip install -r requirements.txt ```

# Dataset
Download [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data)

```
data-|
     |-train-|
             |-0
             |-1
             ...
             |-41
             |-42
                
     |-test -|
             |-000001.png
             |-000002.png
             ...             
     |-Train.csv
     |-Test.csv
    
```

# Train
```python train.py --path ./data  ```


# Evaluation
```python inference.py --path ./data --weights ./weights/best.pt```

# Deployment
1. Installation ONNX RUNTIME C API
``` // Installation ONNX RUNTIME C API
# Step 1
- Download onnxruntime-linux-x64-1.16.3.tgz : https://github.com/microsoft/onnxruntime/releases
- Just unpack files: These are pre-build files, so we do not need any setup processes: onnxruntime-linux-x64-1.16.3.tgz  
 
# Step2
If There are some library path errors:"error while loading shared libraries: libonnxruntime.so.1.16.3: cannot open shared object file: No such file or directory"

Do following commands:
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/german_traffic_sign_classification/deployment/onnxruntime/lib
echo $LD_LIBRARY_PATH 
```

2. Conversion to ONNX

``` python torch2onnx.py --save-onnx ./weights/best.onnx```

3. Run C code

```
g++ -I/onnxruntime/include -o run inference.cc -lonnxruntime -L/onnxruntime/lib `pkg-config --cflags --libs opencv4` -std=c++17

./run 

```

# References
- [ONNXRUNTIME](https://github.com/microsoft/onnxruntime)

# Licence
The project is licensed under the [MIT license](https://github.com/jabborov/german_traffic_sign_classification/blob/main/LICENSE)