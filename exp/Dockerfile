# FROM python:3.10-bookworm

# WORKDIR /code

# # Install system deps
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl openssl \
#     && update-ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip/setuptools first with trusted hosts
# RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip setuptools wheel certifi

# # Copy requirements
# COPY requirements.txt /code/requirements.txt

# # Install requirements with trusted hosts
# RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /code/requirements.txt

# # Install tensorflow-federated explicitly
# RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --quiet --upgrade tensorflow-federated

# RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# COPY . /code

# CMD ["python", "./modifier/fl_model_without_gridsearch.py"]
# CMD ["python", "./modifier/fl_model_fine_tuning_two_phase_train.py"]
# CMD ["python", "./modifier/fl_model.py"]
# CMD ["python", "./app/Client_dataset.py"]
# CMD ["python", "./data/Image_client_classification"]


FROM python:3.10-bookworm

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --upgrade pip setuptools wheel certifi

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --only-binary=grpcio \
    -r /code/requirements.txt

COPY . /code

# CMD ["python", "./modifier/fl_model_without_gridsearch.py"]
# CMD ["python", "./modifier/fl_model_fine_tuning_two_phase_train.py"]
# CMD ["python", "./modifier/fl_model.py"]
# CMD ["python", "./app/Client_dataset.py"]
CMD ["python", "./data/Image_client_classification"]
#=================DockerFile  2 ====================================
# FROM tensorflow/tensorflow:latest-gpu
# FROM tensorflow/tensorflow:2.14.0-gpu 
# FROM tensorflow/tensorflow:2.15.0-gpu

# WORKDIR /code

# # Αφαίρεση της python 3.11
# # RUN apt-get update && apt-get remove -y python3.11 python3.11-minimal
# # 1. ΕΓΚΑΤΑΣΤΑΣΗ Python 3.10 και ρύθμιση ως default

# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3.10-dev \
#     && rm -rf /var/lib/apt/lists/*

#     # 2. ΡΥΘΜΙΣΗ Python 3.10 ως default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2

# # 3. ΕΓΚΑΤΑΣΤΑΣΗ pip για Python 3.10
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl openssl \
#     && update-ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# RUN python3 -m pip install --upgrade pip setuptools wheel
# COPY requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir -r /code/requirements.txt


# COPY . /code

# CMD ["python", "./modifier/fl_model_without_gridsearch.py"]



#=================DockerFile  4 ====================================

# Χρησιμοποιήστε CUDA base image
# FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# WORKDIR /code

# # Εγκατάσταση Python 3.10
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3.10-venv \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# # Εγκατάσταση TensorFlow και requirements
# RUN pip install --upgrade pip
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /code

# CMD ["python", "./modifier/fl_model_without_gridsearch.py"]


# WORKDIR /code

# # 1. ΕΓΚΑΤΑΣΤΑΣΗ system dependencies για TensorFlow
# RUN apt-get update && apt-get install -y \
#     wget \
#     git \
#     build-essential \
#     cuda-command-line-tools-12-0 \
#     libcublas-12-0 \
#     libcudnn8 \
#     && rm -rf /var/lib/apt/lists/*


# # 3. ΕΓΚΑΤΑΣΤΑΣΗ υπολοίπων requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /code

# # 4. ΕΠΙΒΕΒΑΙΩΣΗ
# RUN python --version
# RUN pip list | grep tensorflow

# # 2. ΡΥΘΜΙΣΗ environment variables για GPU
# ENV TF_GPU_THREAD_MODE=gpu_private
# ENV TF_GPU_THREAD_COUNT=2
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}


# CMD ["python", "./modifier/fl_model_without_gridsearch.py"]