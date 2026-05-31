FROM python:3.11.11-bullseye
# FROM python:3.10-bookworm

# Set deterministic environment
ENV PYTHONHASHSEED=0
ENV TF_DETERMINISTIC_OPS=1
ENV TF_CUDNN_DETERMINISTIC=1
# ENV CUDA_VISIBLE_DEVICES=-1

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools wheel
# RUN pip install --upgrade pip "setuptools<70" wheel


# Εγκατάσταση συμβατών εκδόσεων των σπασμένων dependencies
RUN pip install --only-binary=:all: \
    "jax==0.4.17" \
    "jaxlib==0.4.17" \
    "grpcio==1.59.3"

COPY ./requirements.txt /code/requirements.txt

# Install NumPy first with specific version that supports Python 3.11
RUN pip install \
    "attrs>=23.1,<24" \
    "cachetools>=5.3,<6" \
    "absl-py>=1.0,<2" \
    "dm-tree==0.1.8" \
    "dp-accounting==0.4.3" \
    "tensorflow-privacy" \
    "tensorflow-model-optimization" \
    "portpicker" \
    "tqdm" \
    "google-vizier==0.1.11" \
    "tensorflow-compression" \
    # "numpy>=1.22,<2" \
    "scipy" \
    "six"

RUN pip install --no-cache-dir "numpy>=1.22.0"
RUN pip install "setuptools<81.0.0" --upgrade
RUN pip install --no-cache-dir -r /code/requirements.txt

# RUN pip install --upgrade tensorflow
# RUN pip install --quiet --upgrade tensorflow-federated
RUN pip install tensorflow-federated==0.87.0 --no-deps

COPY . /code

CMD ["python", "./app/fl_model_without_gridsearch.py"]
# CMD ["python", "./modifier/fl_model.py"]
