## GTSRB Classification: PyTorch implementation
The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. The properties of GTSRB include:
- Single-image, multi-class classification problem
- 43 classes
- More than 50,000 images in total

## Content Structure
* [Description](#description)
* [Installation](#installation)
* [Dataset](#dataset)
* [Train](#train)
* [Evaluation](#evaluation)
* [Deployment](#deployment)
* [References](#references)
* [Licence](#licence)

# Description
The model is a small CNN with two convolutional blocks (Conv → BatchNorm → ReLU → MaxPool) followed by two fully connected layers. It takes 3×32×32 RGB images normalized to [-1, 1]. The trained PyTorch weights are converted to ONNX and run on CPU from C++ through the ONNX Runtime C API.

### Results (GTSRB test set, 12,630 images)
| Weights | PyTorch (`evaluate.py`) | C++ / ONNX Runtime (`inference.cc`) |
|---|---|---|
| `weights/best.pt` / `best.onnx` (current) | **97.52%** | **97.53%** |
| Previous weights (older, buggy training script) | 94.56% | 95.30% |

Average C++ inference time: about 0.05 ms per image on CPU (batch size 1).

# Installation
Download the project:
```
git clone https://github.com/jabborov/german_traffic_sign_classification.git
cd german_traffic_sign_classification
```

Install the required libraries:
```
pip install -r requirements.txt
```

# Dataset
Download [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data) and extract it into `./gtrsb_dataset` (the default `--path`):

```
gtrsb_dataset/
├── Train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 42/
├── Test/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── Train.csv     # columns include ClassId, Path (e.g. Train/20/00020_00000_00000.png)
└── Test.csv      # columns include ClassId, Path (e.g. Test/00000.png)
```

> Note: the paths in `Train.csv` and `Test.csv` must match the folder names. Some copies of the dataset name the image folders `train_images/` and `test_images/` instead of `Train/` and `Test/`. In that case, create symlinks instead of renaming:
> ```
> cd gtrsb_dataset
> ln -s train_images Train
> ln -s test_images Test
> ```

# Train
```
python train.py --path ./gtrsb_dataset
```
Useful options: `--epochs` (default 100), `--patience` (early stopping, default 15), `--batch-size`, `--lr`, `--workers`, `--seed`. The best weights (by validation accuracy) are saved to `./weights/best.pt`.

# Evaluation
```
python evaluate.py --path ./gtrsb_dataset --weights ./weights/best.pt
```

# Deployment
### 1. ONNX Runtime C API
The prebuilt ONNX Runtime 1.16.3 for Linux x64 is included in `./onnxruntime` (`include/` and `lib/`).
To use another version, download `onnxruntime-linux-x64-<version>.tgz` from the [releases page](https://github.com/microsoft/onnxruntime/releases) and unpack it there. No build step is needed.

### 2. Convert to ONNX
```
python pytorch2onnx.py --weights ./weights/best.pt --save-onnx ./weights/best.onnx
```
After export the script checks the model and compares ONNX Runtime outputs with PyTorch (skip with `--no-verify`).

### 3. Build and run the C++ code
Requires OpenCV 4 (`pkg-config opencv4`) and a C++17 compiler. Run from the project root:
```
g++ -std=c++17 -O2 inference.cc -o run \
    -I./onnxruntime/include -L./onnxruntime/lib -lonnxruntime \
    -Wl,-rpath,'$ORIGIN/onnxruntime/lib' \
    `pkg-config --cflags --libs opencv4`

./run ./weights/best.onnx ./gtrsb_dataset
```
- If the data folder contains `Test.csv`, every listed image is classified and accuracy is reported.
- Otherwise every image in the folder is classified (predictions only).
- Average inference time per image is printed at the end.

The rpath flag lets `./run` find `libonnxruntime.so` without setting `LD_LIBRARY_PATH`. If you build without it and see `error while loading shared libraries: libonnxruntime.so.1.16.3`, run:
```
export LD_LIBRARY_PATH=$PWD/onnxruntime/lib:$LD_LIBRARY_PATH
```

# References
- [ONNXRUNTIME](https://github.com/microsoft/onnxruntime)

# Licence
The project is licensed under the [MIT license](https://github.com/jabborov/german_traffic_sign_classification/blob/main/LICENSE)
