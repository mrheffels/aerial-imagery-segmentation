# Aerial Imagery Pixel-level Segmentation
Codebase for the MSc thesis paper Aerial Imagery Pixel-level Segmentation

## Creating environments for the DroneDeploy fastai/keras benchmark and DeepLabv3+ codebase

It is crucial to take note of your own GPU driver environment. For this work the following environment was available (High Performance Cluster at Eindhoven University of Technology):

- GCC \& G++: 5.4.0 (20160609)
- Nvidia driver: 450.51.06
- CUDA driver: 11.0
- CUDA compilation tools (including nvcc): release 10.2, V10.2.89
- Using TensorFlow through Conda automatically installs the latest CuDNN version in your local environment.

### fastai environment and preparations
This is for use with the dd-ml-benchmark implementation
1. conda create --name fastai\_gpu pytorch torchvision cudatoolkit=10.1 -c pytorch
2. conda activate fastai\_gpu
3. conda install -c fastai fastai=1.0.61
4. conda install opencv typing wandb scikit-learn
5. pip install image-classifiers

### keras environment and preparations
This is for use with the dd-ml-benchmark implementation
1. conda create --name keras\_gpu keras tensorflow-gpu=1.15
2. conda activate keras\_gpu
3. conda install opencv typing wandb scikit-learn
4. pip install image-classifiers

### DeepLabv3+ environment and preparations
This is for use with the models/research/deeplab Tensorflow implementation
1. conda create --name tf1\_gpu tensorflow-gpu=1.15
2. conda activate tf1\_gpu
3. conda install -c conda-forge pillow tqdm numpy
4. pip install tf\_slim
5. From tensorflow/models/research/ directory run: export PYTHONPATH=\$PYTHONPATH:\`pwd\`:\`pwd\`/slim
## List of altered files for DroneDeploy dataset experiments
### DeepLabv3+ Tensorflow research codebase {https://github.com/tensorflow/models/tree/master/research/deeplab} extensions
- convert\_rgb\_to\_index.py (altered to strip 3 dimensional segmentation labels to 1 dimensional)
- build\_dd\_data.py (altered for DroneDeploy compatiblity)
- data\_generator.py (altered for DroneDeploy compatiblity)
- train-dd-full.sh, eval-dd.sh, vis-dd.sh (dataset adaptations inspired by this GitHub repo\footnote[2]{https://github.com/heaversm/deeplab-training})

### DroneDeploy benchmark codebase {https://github.com/dronedeploy/dd-ml-segmentation-benchmark} extensions
- custom\_training.py (implementation Focal loss function for fastai u-net)
- custom\_training\_keras.py (implementation Focal loss function for Keras u-net)
- images2chips.py (added test images conversion to tiles for DeepLabv3+ compatibility)
- scoring.py (added mean IOU and IOU score per class metric)
