# Use CUDA 11.4 base image
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /workspace

# Install Python 3.8 and pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    build-essential \
    git \
    && apt-get clean

# Make python3.8 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install \
        "numpy<=1.23.5" \
        "protobuf<=3.20.3" \
        pandas \
        scikit-learn \
        tensorflow==2.6.2 \
        shap \
        matplotlib \
        logomaker \
        imbalanced-learn

# Default command
CMD ["/bin/bash"]