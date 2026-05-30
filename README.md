Create the cirtual environment:  python -m venv env

Activate the virtual environment: env\Scripts\activate

Install the requiremets.txt: pip install -r requirements.txt

Deactivate the virtual environment: deactivate

# Mount local dir to /plots in container
docker build -t thesis-fl . && docker run -v "%cd%/modifier/plots:/code/modifier/plots"  -v "%cd%/modifier/lr_auc_results:/code/modifier/lr_auc_results" thesis-fl

docker build -t thesis-fl-without-grid-search . && docker run --gpus all -v "%cd%/modifier/plot_without_gridsearch:/code/modifier/plots" thesis-fl-without-grid-search

docker build -t thesis-fl-without-grid-search . && docker run --runtime=nvidia -v "%cd%/modifier/plot_without_gridsearch:/code/modifier/plots" thesis-fl-without-grid-search

docker build -t thesis-fl-without-grid-search . && docker run --gpus all -v "%cd%/modifier/plot_without_gridsearch:/code/modifier/plots"  -v "%cd%/data/classification/preprocessedRF:/code/data/classification/preprocessedRF"  -v "%cd%/data/augmented:/code/data/augmented/"  -v "%cd%/modifier/trained_model_weights:/code/modifier/trained_model_weights" thesis-fl-without-grid-search

# Only for prediction with saved model
docker build -t thesis-prediction-model . && docker run -v "%cd%\modifier\trained_model_weights:/code/modifier/trained_model_weights" thesis-prediction-model

docker build -t fine-tuning-fl-without-grid-search . && docker run -v "%cd%/modifier/plot_without_gridsearch:/code/modifier/plots" -v "%cd%/data/augmented:/code/data/augmented/" fine-tuning-fl-without-grid-search

docker build -t client_dataset . && docker run -v "%cd%/data/classification/preprocessedRF:/code/data/classification/preprocessedRF" client_dataset

docker build -t save_augmented_dataset . && docker run -v "%cd%/data/federated_images_with_batches:/code/data/federated_images_with_batches/" save_augmented_dataset

docker build -t dataset . && docker run -v "%cd%/data/tff_dataset:/code/data/tff_dataset"

docker build -t dataset_exp . && docker run dataset_exp


# With powershell
docker build -t thesis-fl . ; docker run -v "%PWD%/modifier/plots:/code/plots" thesis-fl
# With linux 
docker build -t thesis-fl . && docker run -v "%pwd%/modifier/plots:/code/plots" thesis-fl

# run docker compose and rebuild the docker image
docker up --build

wsl -t Ubuntu
wsl --shutdown

