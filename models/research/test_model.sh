#!/bin/bash
source /home/TUE/20176671/miniconda3/etc/profile.d/conda.sh
source activate tf1_gpu
python3 deeplab/model_test.py
tensorboard --logdir saved
source deactivate
