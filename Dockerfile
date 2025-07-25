# Base container that includes all dependencies but not the actual repo
# Updated from templates in the [softlearning (SAC) library](https://github.com/rail-berkeley/softlearning)

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04 as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Set environment variables
ENV MINICONDA_HOME /opt/conda
ENV PATH=$MINICONDA_HOME/bin:$PATH

# Install necessary build tools and download Miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
# Always check repo.anaconda.com/miniconda for the latest installer link
# Use -b for batch mode (no prompts) and -p for the installation prefix
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $MINICONDA_HOME && \
    rm miniconda.sh

# Initialize Conda for use in the container.
# This makes `conda activate` work correctly.
# 'base' is the default environment that Miniconda creates.
RUN conda init bash && \
    conda clean --all -f -y

RUN conda config --remove channels CHANNEL

RUN conda create --name roble python=3.10 pip
RUN echo "source activate roble" >> ~/.bashrc
## Make it so you can install things to the correct version of pip
ENV PATH /opt/conda/envs/roble/bin:$PATH
RUN source activate roble

# Set the working directory for your application
WORKDIR /mini_grp

COPY . /app
RUN ls
## Install the requirements for your learning code.
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

## Install pytorch and cuda
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check that the sime loads and pre-build mujoco in the docker image. Better to catch these errors here.
RUN python -c "import gym; env = gym.make('Ant-v2'); print(env)"
# COPY you code to the docker image here.
# e.g.
ADD conf conf
ADD mini_grp2.py mini_grp2.py

## Check the file were copied
RUN ls
