# Start from the mosaicml/composer base image
FROM mosaicml/composer:0.29.0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl wget git vim sudo nvidia-settings && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y xserver-xorg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone AI2THOR Docker repository and install NVIDIA dependencies
RUN git clone https://github.com/allenai/ai2thor-docker.git && \
    sh ai2thor-docker/scripts/install_nvidia.sh && \
    rm -rf ai2thor-docker

# Set up Conda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

RUN bash miniconda.sh -b -p /root/miniconda3 && \
    rm miniconda.sh

RUN /root/miniconda3/bin/conda init bash
RUN echo "source /root/miniconda3/bin/activate" >> ~/.bashrc

# Create AI2THOR environment with Python 3.6
RUN /root/miniconda3/bin/conda create -y -n ai2thor_env python=3.6 && \
    echo "conda activate ai2thor_env" > /root/activate_ai2thor.sh && \
    /root/miniconda3/bin/conda clean -ya

# Install AI2THOR in the Python 3.6 environment
SHELL ["/bin/bash", "-c"]
RUN source /root/miniconda3/bin/activate ai2thor_env && \
    pip install --ignore-installed ai2thor==2.1.0 flask requests opencv-python-headless==4.5.3.56 pillow numpy pandas networkx h5py tqdm vocab revtok boto3

# Install PyTorch nightly and additional packages in the base environment
RUN source /root/miniconda3/bin/activate base && \
    pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall && \
    pip install --upgrade torchvision && \
    pip install huggingface_hub datasets transformers==4.49.0 accelerate && \
    pip install requests pillow awscli fuzzywuzzy word2number

# Set up work directory
WORKDIR /