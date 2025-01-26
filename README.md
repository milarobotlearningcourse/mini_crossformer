[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/milarobotlearningcourse/mini-grp/blob/main/mini-grp.ipynb)

# Mini GRP: Mini- Generalist Robotics Policy

Minimialist reimplimentation of the Octo Generalist Robotics Policy.

## Install

'''module load cudatoolkit/11.8 miniconda/3'''

conda create -n mini-grp python=3.10
conda activate mini-grp
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torch==2.4.0
pip install hydra-submitit-launcher --upgrade

### Install MilaTools

pip install milatools==0.1.14 decorator==4.4.2 moviepy==1.0.3

## Dataset

https://rail-berkeley.github.io/bridgedata/

## Install SimpleEnv

Prerequisites:

    CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
    An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Clone this repo:

```
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
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

conda install conda-forge::vulkan-tools conda-forge::vulkan-headers


## Running the code

Basic example to train the GRP over the bridge dataset 

```
python mini-grp.py
```

Launch multiple jobs on a slurm cluster to evalute different model architectures, etc.
```
python mini-grp.py --multirun gradient_accumulation_steps=1,2,4 hydra/launcher=submitit_slurm
```


### License

MIT
