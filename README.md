# A Federated Learning Approach for Privacy-Preserving Automated Signature Verification

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![TensorFlow Federated](https://img.shields.io/badge/TensorFlow_Federated-latest-orange.svg)](https://www.tensorflow.org/federated)
[![Docker](https://img.shields.io/badge/Docker-required-blue.svg)](https://www.docker.com/)

Official implementation of the paper: **"A Federated Learning Approach for Privacy-Preserving Automated Signature Verification"**.

📄 **Citation**

```bibtex
@inproceedings{harisv2026federated,
  title={A Federated Learning Approach for Privacy-Preserving Automated Signature Verification},
  author={V., Haris and [Other Authors]},
  year={2026},
  url={https://scholar.google.com/citations?view_op=view_citation&hl=el&user=a9kHE_YAAAAJ&citation_for_view=a9kHE_YAAAAJ:hFOr9nPyWt4C}
}
```

---

## 📌 Abstract
The growing interconnectivity of digital systems has led to the massive collection and
centralization of sensitive data, raising serious concerns about confidentiality and compli-
ance with privacy regulations. Biometric authentication systems, such as offline signature
verification, are particularly vulnerable. Federated learning (FL) provides a promising
framework by enabling model training without exposing raw client data. However, keep-
ing data strictly localized inherently creates severe data scarcity, which is a significant
barrier to building robust deep learning (DL) models. This work investigates the feasibility
of a privacy-preserving writer-dependent (WD) offline signature verification (OSV) system
within an FL framework. To make local training viable under these constraints, we integrate
complementary techniques into the federated pipeline: data augmentation is utilized to
increase local sample diversity, while transfer learning provides robust pre-trained fea-
ture representations, drastically reducing the volume of data required for effective local
fine-tuning. The proposed WD-OSV system was trained and evaluated on the popular
CEDAR signature dataset, for which an average area under the curve of 0.8893, along with
an average binary accuracy (ACC) of 80.12%, are reported as preliminary results.

---

## 🏗️ Architecture

The system follows a standard Federated Learning architecture using the **FedAvg** algorithm, designed to train a robust Automated Signature Verification model while keeping local signature data decentralized.

```
       [ Central Server ]  <-- (Aggregates Weights via FedAvg)
         /      |      \
     Push      Push     Push  (Global Model Weights)
       /        |        \
      v         v         v
  [Client 1] [Client 2] [Client N]  <-- (Local Training on Private Signatures)
```

---

## 📁 Project Structure
```
.
├── app/
│   ├── Client_dataset.py          # Dataset loading, preprocessing, and train/eval/test splitting
│   ├── Image_classification.py    # Raw image sorting by client ID and random forgery assignment
│   ├── models.py                  # Model definitions (DenseNet121, ResNet50V2, EfficientNetV2B0, MobileNetV3Small, custom CNNs)
│   ├── utils.py                   # Metrics, plotting (AUC, EER, FAR/FRR), model saving, metadata
│   ├── train_fed.py               # Main federated training loop with early stopping and evaluation
│   ├── grid_search_lr.py          # Learning rate grid search script
│   └── load_tff_dataset.py        # TFF dataset loader helper
├── data/
│   └── signatures/                # <-- Place raw CEDAR .png images here before running Docker
├── Dockerfile
└── docker-compose.yml
```

---

## 📊 Dataset Setup 
This project uses the CEDAR offline signature dataset. The dataset is not included in this repository. Download it from the [official source](http://www.cedar.buffalo.edu/NIJ/data/).

### Expected filename format

The data pipeline expects images to follow these naming conventions:

- Genuine signatures: `original_<client_id>_<image_id>.png`
- Skilled forgeries: `forgeries_<client_id>_<image_id>.png`


---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) (required — all paths are configured for Linux containers)


### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/Haris-v/A-Federated-Learning-Approach-for-Privacy-Preserving-Automated-Signature-Verification.git
cd A-Federated-Learning-Approach-for-Privacy-Preserving-Automated-Signature-Verification
```

**2. Create the required local directories:**

These folders must exist on the host machine before starting Docker so that the volume mounts work correctly and output files (models, plots, augmented images) are accessible locally after training.

```bash
mkdir -p data/signatures
mkdir -p data/classification/signatures
mkdir -p data/classification/signaturesRF
mkdir -p data/augmented
mkdir -p results/trained_model_weights
mkdir -p results/plots
mkdir -p results/lr_auc_results
```

**3. Download the CEDAR dataset and place the raw images:**

Download the CEDAR signature dataset from the [official source](http://www.cedar.buffalo.edu/NIJ/data/) and place all raw `.png` files directly inside `data/signatures/`. Filenames must follow the expected naming convention (see [Dataset Setup](#-dataset-setup)).


**4. Verify your `docker-compose.yml` includes the volume mounts:**

The following mounts are required so that the container can read the input data and write outputs back to your local machine:

```yaml
volumes:
      - ./data/signatures:/code/data/signatures
      - ./data/classification/signatures:/code/data/classification/signatures
      - ./data/classification/signaturesRF:/code/data/classification/signaturesRF
      - ./data/classification/preprocessedRF:/code/data/classification/preprocessedRF
      - ./data/augmented:/code/data/augmented
      - ./results/plots:/code/results/plots
      - ./results/trained_model_weights:/code/results/trained_model_weights
      - ./results/lr_auc_results:/code/results/lr_auc_results
```

**5. Configure the Execution Mode (Dockerfile CMD):**

Before building and running your Docker container, you must configure the CMD directive at the bottom of your Dockerfile depending on your execution goals:

  - To run the project WITH Grid Search (Hyperparameter Tuning): Open your Dockerfile and ensure the CMD points to the grid search script:
  ```
  CMD ["python", "./modifier/fl_model.py"]
```

- To run the project NORMALLY (Standard Federated Training): If you want to train the model with fixed hyperparameters, update the CMD to point to the standard script:

```
CMD ["python", "./app/fl_model_without_gridsearch.py"]
```

  **6. Build and start the Docker container:**

```bash
docker-compose up --build
```
