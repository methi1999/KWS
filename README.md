# Introduction
Keyword Spotting (KWS) refers to the task of detecting a pre-defind keyword/phrase in an audio file or a stream of audio. The implemented algorithm uses a sliding Dynamic Time Warping (DTW) approach. Refer to [this paper](https://ieeexplore.ieee.org/abstract/document/6140822/) for a detailed explanation. You can also view my presentation [here](https://methi1999.github.io/pdf/kws.pdf).

# Datasets

1. [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) is used for training a Neural Network which acts as a feature extractor.
2. The [Google Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) dataset is used for testing the performance of the algorithm.

# Instructions

1. The following python packages are required: numpy, matplotlib, pickle, torch, json, scipy, python_speech features, yaml
2. For relative paths to work smoothly, please adhere to the following directory structure:

```bash
KWS (parent directory)
├── speech (Google Speech Commands)
│	├── bed (example class)
│	├── ...
├── nn
│	├── TIMIT
│	├── TEST
│	├── TRAIN
│	├── models (where trained models are stored)
│	│	├── best.pth (a shallow pre-trained model with ±4 context is included)
│	│	├── (other models)
│	├── (python scripts and config file)
```

3. 'dl_model.py' is responsible for training the Neural Network feature extractor while 'sliding_kws.py' runs the actual experiments and dumps a json file containing the results.