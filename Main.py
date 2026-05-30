from modifier.Image_client_classification import Image_classification
from modifier.Client_dataset import Dataset

print("The image sorting by client id is started...")
print("")
clasify_raw_data = Image_classification()
clasify_raw_data.sort_images_by_clientid()
print("")
print("The image sorting by client id is stoped...")
print("")

get_dataset = Dataset()
train_dataset, test_dataset = get_dataset.load_dataset()
