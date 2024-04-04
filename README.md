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
* [Licence](#licence)
* [Reference](#reference)
# Description
- CNN 2 layers used to achieve high accuracy. My approach acchieved the accuracy of 90.0%.

# Installation
Download the project:
```git clone```

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

# Deployment for CPU based edge devices
2. Installation ONNXRUNTIME

2. Conversion to ONNX
``` python torch2onnx.py ```

3. Run 


# References

# Licence
The project is licensed under the [MIT license](LICENSE)