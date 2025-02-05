

# HW: Making a Generalist Robotics Policy

A Generalist Robtoics Policy (GRP) is made up from a modified [vision transformer](https://arxiv.org/abs/2010.11929). A vision transfer is a modified version of a transformer that is designed to process images instead of text. In order for a transformer to process images the images need to be sliced up into patches that can be tokenized.

You can complete the homework by addressing the todos in [mini-grp.py](mini-grp.py)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/milarobotlearningcourse/robot_learning_2025/blob/main/hw2/mini-grp-learn.ipynb)

## Installation

```
conda create -n mini-grp python=3.10
conda activate mini-grp
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torch==2.4.0
pip install hydra-submitit-launcher --upgrade
```

### Install SimpleEnv

Prerequisites:

    CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
    An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Clone this repo:

```
git clone https://github.com/milarobotlearningcourse/SimplerEnv --recurse-submodules
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):

```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:

```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:

```
cd {this_repo}
pip install -e .
```

```
conda install conda-forge::vulkan-tools conda-forge::vulkan-headers
```

# GRP Transformer Similar to [Octo](https://octo-models.github.io/).

The provided [code](mini-grp.py) is an example of a vision transformer. Modifiy this code to become a multi-modal transformer model that accepts images and text as input and outputs either classes or continuous values. Make sure to impliment the block masking to train the model to work when goals are provided via images or text.

### Discrete vs Continuous Action Space

There are different methods that can be used to model the action distribution. Many papers have discretetized the action space (OpenAI Hand), resulting in good performance. Train the GRP model with a discrete representation (cross entropy) vs a continuous representation (MSE) and compare the performance of these two distributions. Compare the performance in [simpleEnv](https://github.com/milarobotlearningcourse/SimplerEnv)

### Effect of Encoding Size

In this section the goal is compare training results when using two different encoding sizes 128 vs 256. Remark on the performance difference when using these two different encoding sizes. 

### Replace the Text Encoder with the one from T5

The text tokenization and encoding provided in the initial version of the code is very simple and basic. It may be possible to improve goal generalizing by improving the tokenization used. Use the tokenizer from the [T5 model](https://jmlr.org/papers/v21/20-074.html) to tokenize the text used to encode the goal descriptions. Some example code to get started is available [here](https://huggingface.co/docs/transformers/en/model_doc/t5).


## Grow the Dataset

The dataset used for training is rather small (100 trajectories) but works and fits on small GPUs. Use the [create_mini_oxe_dataset.py](create_mini_oxe_dataset.py) file to collect more data (250 trajectories instead of 100) and retrain the model. Does performance increase? Share the learning curves.

## State History

For most robotics problems a single image is not enough to determine the dyanmics well enough to predict the next state. This lack of dynamics information means the model can only solve certain tasks to a limited extent. To provide the model with sufficient state information update the GRP input to include 2 images from the state history and evalaute the performance of this new model. Remark on the change in performance.

## Action Chunking

One of the many methods used to smooth motion and compensate for multi-modal behaviour in the dataset is to predict many actions at a time. 

## Tips:

1. If you are having trouble training the model and not running out of memory, use a smaller batch size and gradient accumulation. Training will take a little longer, but should work.

