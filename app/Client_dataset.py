import os
from sys import prefix
import PIL.ImageShow
# from modifier.save_read_tff_Dataset import save_federated_dataset
from save_read_tff_Dataset import save_federated_dataset
# from save_read_tff_Dataset import save_federated_dataset, save_single_federated_dataset
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import random 
from collections import OrderedDict
from matplotlib import pyplot as plt
import math
import PIL
import collections
from collections import defaultdict



class Dataset():
    DATA_CLASSIFICATION_WINDOWS_FOLDER_PATH = Path(r"..\data\classification\signatures")
    DATA_CLASSIFICATION_SIGNATURE_RF_WINDOWS_FOLDER_PATH = Path(r"..\data\classification\signaturesRF")

    DATA_PREPROCESSED_CLASSIFICATION_FOLDER_PATH = Path(r"..\data\classification\preprocessed")
    # DATA_PREPROCESSED_SIGNATURE_RF_CLASSIFICATION_FOLDER_PATH = Path(r"..\data\classification\preprocessedRF")
    # DATA_PREPROCESSED_SIGNATURE_CLASSIFICATION_FOLDER_PATH = Path(r"/code/data/classification/preprocessed")
    DATA_PREPROCESSED_SIGNATURE_RF_CLASSIFICATION_FOLDER_PATH = Path(r"/code/data/classification/preprocessedRF")   

    DATA_AUGMENTED_SIGNATURE_RF_FOLDER_PATH = Path(r"/code/data/augmented") #not used
    #Path for docker.
    DATA_CLASSIFICATION_DOCKER_FOLDER_PATH = Path("/code/data/classification/signatures")
    DATA_CLASSIFICATION_SIGNATURE_RF_DOCKER_FOLDER_PATH = Path("/code/data/classification/signaturesRF")



    PNG_FILE_PREFIX = ".png"
    seed_collection = (15, 64, 5, 65, 35, 23, 19, 7, 48, 33, 77, 99, 0, 6, 27, 40, 38, 58, 62, 67, 72, 76, 83, 86, 91,
                       10, 18, 11, 21, 26, 31, 39, 73, 61, 68, 80, 87, 93, 94, 2, 12, 16, 20, 36, 53, 81, 43, 69, 74, 34,
                       88, 92, 4, 60, 29)
    
       
    def __init__(self, path = None):
        self.path = path

    def load_dataset(self):

        seed_counter = 0
        train_client_dataset={}
        eval_client_dataset={}
        test_client_dataset={}
        test_client_dataset_without_skilled={}

        if self.path is None:
            # self.path = self.DATA_CLASSIFICATION_DOCKER_FOLDER_PATH
            self.path = self.DATA_CLASSIFICATION_SIGNATURE_RF_DOCKER_FOLDER_PATH 

        
        for root, dirs, files in os.walk(self.path):
            random.seed(self.seed_collection[seed_counter])

            if len(files) > 0 and self.is_image(files[0]):
                # Split the filenames into training and testing datasets for spliting
                # the dataset into training and testing according the filenames
                # train_filenames, eval_filenames, test_filenames = self.get_train_and_eval_filenames(files)
                train_filenames, eval_filenames, test_filenames, test_filenames_without_skilled = self.get_train_and_eval_filenames(files)
                
                client_id_name = self.get_client_id(root)

                train_client_dataset[client_id_name] = self.create_dataset(root, train_filenames, False)
                eval_client_dataset[client_id_name] = self.create_dataset(root, eval_filenames, False)
                test_client_dataset[client_id_name], test_dataset_labels = self.create_dataset(root, test_filenames, True)
                test_client_dataset_without_skilled[client_id_name], test_labels_without_skilled = self.create_dataset(root, test_filenames_without_skilled, True)

                print("------------------------------------")
                print(f"Keys of dataset")
                print(train_client_dataset.keys())
                print("---------------------------------------")

                seed_counter += 1
   
        #  Paths for saving the dataset into tff.records that is tf.data.Dataset (the federated dataset) instead the images
        to_save_train_data_path = Path(r"..\data\tff_dataset\train")
        to_save_eval_data_path = Path(r"..\data\tff_dataset\eval")
        #
        # Save the tf.data.Dataset dataset in tf.records format
        # save_federated_dataset(train_client_dataset, to_save_train_data_path)
        # save_federated_dataset(eval_client_dataset, to_save_eval_data_path)
        # save_federated_dataset(test_client_dataset, to_save_test_data_path)

        return train_client_dataset, eval_client_dataset, [test_client_dataset, test_dataset_labels], [test_client_dataset_without_skilled, test_labels_without_skilled]
 


    def get_train_and_eval_filenames(self, files):
        original_list = []
        random_forgery_list = []
        forgery_list = []
        eval_list = []
        test_list = []

        for file in files:
            splitted_files = file.split("_")[0]

            if splitted_files == "original":
                original_list.append(file)
            elif splitted_files == "random":
                random_forgery_list.append(file)
            else:
                forgery_list.append(file)

        
        original_subset = round(0.8 * len(original_list))
        random_forgery_subset = round(0.8 * 24)
        
        #Training dataset
        train_dataset_files = original_list[:original_subset]
        train_dataset_files.extend(random_forgery_list[:random_forgery_subset])

        #Evaluation Dataset
        eval_list = original_list[original_subset:round(0.9 * len(original_list))]
        eval_list.extend(random_forgery_list[random_forgery_subset:round(0.9*24)])

        #Test dataset with skilled forgeries
        # test_list = forgery_list[:10]
        test_list = forgery_list[:4]
        test_list.extend(original_list[round(0.9 * len(original_list)):])
        test_list.extend(random_forgery_list[round(0.9*24) : 24])

        #Test dataset without skilled forgeries
        test_without_skilled = original_list[round(0.9 * len(original_list)):]
        test_without_skilled.extend(random_forgery_list[round(0.9*24) : 24])

        print("")
        print("")
        print("Number of signatures per client Before split")
        print("")
        print(f"Original: {len(original_list)}")
        print(f"Random Forgeries: {len(random_forgery_list)}")
        print(f"Forgeries: {len(forgery_list)}")
        print("")
        print("=========================================================")
        print("")
        print("=========================================================")
        print("Number of signatures per client After split")
        print("")
        print(f"train dataset: {len(train_dataset_files)}")
        print(f"Evaluation dataset: {len(eval_list)}")
        print(f"Test dataset: {len(test_list)}")
        print("")
        print("=========================================================")
        print("")
        print("=========================================================")
        print("")
        print("")
        print("Items of each dataset")
        print("")
        print(f"Training dataset:  {train_dataset_files}")
        print(f"Evaluation dataset:  {eval_list}")
        print(f"Test dataset:  {test_list}")

        return train_dataset_files, eval_list, test_list, test_without_skilled

 

    def find_maximum_dimensions_of_images(self, path):
        width_max = 0
        height_max = 0

        for root, dirs, files in os.walk(path):
            for item in files:
                if self.is_image(item):
                    
                    # For windoes path
                    # img = Image.open(root + "\\" + item)

                    # For linux - docker path
                    img = Image.open(root + "/" + item)
                    width_img, height_image = img.size
                    img.close()

                    if width_max < width_img:
                        width_max = width_img
                    
                    if height_max < height_image:
                        height_max = height_image
        
        return width_max, height_max


    def is_image(self,item):
        if item.endswith(self.PNG_FILE_PREFIX):
            return True
        else:
            return False


    def create_dataset(self, root, files, is_test_dataset):

        label_append = []
        pixels_append = []
        counter = 0

        for item in files:
            if self.is_image(item):
                counter +=1

                img = Image.open(os.path.join(root, item))
                # image_name = img.filename.split("\\")[-1]
                image_name = img.filename.split("/")[-1]

                image = ImageOps.grayscale(img)
                # Resize the image with the same aspect ratio
                resized_image = self.resize_image_axis(224, 224, image)
                background_processed_image = self.background_improvement(resized_image,  image_name=image_name)
                # self.save_preprocessed_images(background_processed_image, self.DATA_PREPROCESSED_CLASSIFICATION_FOLDER_PATH, img.filename)  
                self.save_preprocessed_images(background_processed_image, self.DATA_PREPROCESSED_SIGNATURE_RF_CLASSIFICATION_FOLDER_PATH, img.filename) 
                img_numpy_array = self.convert_image_to_numpy_array(background_processed_image)
                img.close()

                if img_numpy_array.ndim == 2:
                    img_numpy_array = np.expand_dims(img_numpy_array, axis=-1)
                
                tensor, label = self.get_tensor_and_label(item, img_numpy_array)   

                label_append.append(label)
                pixels_append.append(tensor)

        dataset = tf.data.Dataset.from_tensor_slices({
            "pixels": tf.stack(pixels_append),
            "label": tf.stack(label_append)
        })
        if is_test_dataset:
            test_dataset_labels = OrderedDict({
                "labels": tf.stack(label_append)
            })
            return dataset, test_dataset_labels
        else:
            return dataset



    def resize_image_axis(self, width_max, heigth_max, image):

        image_width, image_heigth = image.size
        # Calculate the scaling factor for expanding the image whith the same aspect ratio
        scaling_factor = min(width_max / image_width, heigth_max / image_heigth)
        # Calculate the axis for resizing the image
        modified_width = int(image_width * scaling_factor)
        modified_height = int(image_heigth * scaling_factor)
        resized_image = image.resize((modified_width, modified_height), Image.Resampling.LANCZOS)

        # Calculate the padding for center location
        background_image = Image.new("L", (width_max, heigth_max), 255)
        x_padding = math.floor((width_max - modified_width) / 2)
        y_padding = math.floor((heigth_max - modified_height) / 2 )
        
        background_image.paste(resized_image, (x_padding, y_padding))

        return background_image   
    
    def convert_image_to_numpy_array(self, img):
        return np.array(img) / 255.0


    def get_client_id(self,root):
        # root_parsed_windows = root.split("\\")
        root_parsed_docker = root.split("/")
        # client_id_name = root_parsed_windows[len(root_parsed_windows)-1]
        client_id_name = root_parsed_docker[len(root_parsed_docker)-1]
        return client_id_name


    def get_tensor_and_label(self, item, img_numpy_array):
        tensor = tf.constant(img_numpy_array, dtype=tf.float32)
        label_name = item.split("_")[0]
        if label_name == "original":
            label = tf.constant(1, dtype=tf.int32)
        elif label_name == "forgeries" or label_name == "random":
            label = tf.constant(0, dtype=tf.int32)
        return tensor, label


    def get_element_spec(self):
        element_spec = OrderedDict()
        element_spec["pixels"] = tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        element_spec["label"] = tf.TensorSpec(shape=(), dtype=tf.int32)
        return element_spec

    
    def save_preprocessed_images(self, img, to_root_folder, img_filename):
        # client_folder = img_filename.split("\\")[-2]
        client_folder = img_filename.split("/")[-2]

        image_save_folder_path = to_root_folder.joinpath(client_folder)
        if not os.path.exists(image_save_folder_path):
            os.makedirs(image_save_folder_path)
                
        if not os.path.exists(image_save_folder_path / Path(img_filename).name):
            img.save(image_save_folder_path / Path(img_filename).name)
            print("Image saved succesfully: ", img_filename)
            print("To folder: ", image_save_folder_path)
        else:
            print("The file is already exists!")
            print("To folder", image_save_folder_path)


    def background_improvement(self, img: Image, image_name):
        background_max = 220   # Example: dark gray or black
        # background_min = 200 # Example: lighter gray
        # background_max = 200   # Example: dark gray or black
        # background_min = 180 # Example: lighter gray

        image_array = np.array(img)
        type_of_img = image_name.split("_")[0]

        if type_of_img == "original" or type_of_img == "forgeries" or type_of_img == "random":
            
            for x in range(img.height):
                for y in range(img.width):
                    # if background_min <= image_array[x, y] and image_array[x, y] <= background_max:
                    #     image_array[x, y] = 255  # Replace with white (max grayscale value)
                    
                    if background_max <= image_array[x, y]:
                       image_array[x, y] = 255  # Replace with white (max grayscale value)
                    # elif image_array[x, y] >= background_min:
                    #     image_array[x, y] = 0

            output_image = Image.fromarray(image_array)
            return output_image


# data = Dataset()
# data.load_dataset()
'''

# Second
def apply_augmentation_pipeline_simple(dataset, base_filename, client_id, num_augm=1):
    """
    Απλοποιημένη έκδοση που αποθηκεύει πρώτα τις original 
    και μετά τις augmented με ξεχωριστά ονόματα
    """
    num_augmentations = num_augm
    
    # Πρώτα, κάνε iterate το dataset και αποθήκευσε τα original
    save_dir = Path(base_filename).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Συλλογή όλων των δεδομένων σε λίστα
    all_samples = []
    
    print(f"\n🔧 Processing client {client_id}...")
    
    # 1. Μετατροπή dataset σε λίστα και αποθήκευση originals
    original_samples = []
    sample_counter = 0
    
    for element in dataset:
        pixels = element['pixels']
        label = element['label']
        
        # Αποθήκευση ως numpy για ευκολία
        pixels_np = pixels.numpy()
        label_np = label.numpy()
        
        # Δημιουργία filename για original
        filename = f"{base_filename}_original_sample{sample_counter:04d}_client{client_id}_label{label_np}.png"
        
        # Αποθήκευση εικόνας
        img_array = (pixels_np * 255).astype(np.uint8)
        Image.fromarray(img_array.squeeze(), mode='RGB').save(filename)
        
        original_samples.append({
            'pixels': pixels_np,
            'label': label_np,
            'filename': filename,
            'index': sample_counter
        })
        
        # Προσθήκη στα συνολικά samples
        all_samples.append({
            'pixels': pixels_np,
            'label': label_np,
            'is_original': True,
            'aug_index': -1,
            'sample_index': sample_counter,
            'client_id': str(client_id),
            'filename': filename
        })
        
        sample_counter += 1
    
    print(f"  ✓ Saved {len(original_samples)} original images")
    
    # 2. Δημιουργία και αποθήκευση augmented εικόνων
    for aug_idx in range(num_augmentations):
        print(f"  Creating augmentation iteration {aug_idx}...")
        
        for i, original in enumerate(original_samples):
            pixels = original['pixels']
            label = original['label']
            
            # Convert to tensor for augmentations
            pixels_tensor = tf.constant(pixels, dtype=tf.float32)
            
            # Εφαρμογή augmentations
            angle = tf.random.stateless_uniform(
                [], 
                seed=(42 + aug_idx, i), 
                minval=-1.5708,#-0.26, 
                maxval=1.5708 #0.26
            )
            
            pixels_aug = tfa.image.rotate(
                pixels_tensor,
                angle,
                interpolation='NEAREST',
                fill_mode='constant',
                fill_value=1.0
            )
            
            pixels_aug = tf.image.stateless_random_contrast(
                pixels_aug, 
                0.5, 
                1.8, 
                seed=(100 + aug_idx, i)
            )

            # 3. ΠΡΟΣΘΗΚΗ RANDOM BRIGHTNESS
            pixels_aug = tf.image.stateless_random_brightness(
                pixels_aug,
                max_delta=0.2,  # Έντονη αλλαγή φωτεινότητας
                seed=(200 + aug_idx, i)
            )
            
            pixels_aug = tf.clip_by_value(pixels_aug, 0.0, 1.0)
            pixels_aug_np = (pixels_aug.numpy() * 255).astype(np.uint8)
            
            # Δημιουργία filename
            filename = f"{base_filename}_aug{aug_idx:02d}_sample{i:04d}_client{client_id}_label{label}.png"
            
            # Αποθήκευση
            Image.fromarray(pixels_aug_np.squeeze(), mode='RGB').save(filename)
            
            # Προσθήκη στα συνολικά samples
            all_samples.append({
                'pixels': pixels_aug_np / 255.0,  # Normalize back to [0, 1]
                'label': label,
                'is_original': False,
                'aug_index': aug_idx,
                'sample_index': i,
                'client_id': str(client_id),
                'filename': filename
            })
        
        print(f"    Created {len(original_samples)} augmented images for iteration {aug_idx}")
    
    # 3. Δημιουργία TensorFlow dataset από τα samples
    print(f"  Total images created: {len(all_samples)}")
    print(f"    - Original: {len(original_samples)}")
    print(f"    - Augmented: {len(all_samples) - len(original_samples)}")
    
    # Δημιουργία tensors
    pixels_list = np.array([sample['pixels'] for sample in all_samples])
    labels_list = np.array([sample['label'] for sample in all_samples])
    is_original_list = np.array([sample['is_original'] for sample in all_samples])
    aug_index_list = np.array([sample['aug_index'] for sample in all_samples])
    sample_index_list = np.array([sample['sample_index'] for sample in all_samples])
    client_id_list = [sample['client_id'] for sample in all_samples]
    filename_list = [sample['filename'] for sample in all_samples]
    
    dataset = tf.data.Dataset.from_tensor_slices({
        'pixels': tf.constant(pixels_list, dtype=tf.float32),
        'label': tf.constant(labels_list, dtype=tf.int32),
        'is_original': tf.constant(is_original_list, dtype=tf.bool),
        'aug_index': tf.constant(aug_index_list, dtype=tf.int32),
        'sample_index': tf.constant(sample_index_list, dtype=tf.int32),
        'client_id': tf.constant(client_id_list, dtype=tf.string),
        'filename': tf.constant(filename_list, dtype=tf.string)
    })
    
    return dataset

NUM_CLIENTS = 30
NUM_EPOCHS = 6
BATCH_SIZE =2
SHUFFLE_BUFFER = 1000
PREFETCH_BUFFER = tf.data.AUTOTUNE
NUM_ROUNDS = 400
NUM_AUGMENTATION=4

test_num_data = 0

def preprocess(dataset, client_id, is_training=False):
    global test_num_data

    def convert_to_rgb(element):
        pixels = element['pixels']
        label = element['label']
        pixels = tf.image.grayscale_to_rgb(pixels)
        pixels = tf.ensure_shape(pixels, [224, 224, 3])
        
        return OrderedDict(pixels=pixels, label=label)
    
    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 224, 224, 3]),
            y=tf.reshape(element['label'], [-1, 1]),
            is_original=tf.reshape(tf.cast(element.get('is_original', False), tf.int32), [-1, 1]),
            aug_index=tf.reshape(element.get('aug_index', -1), [-1, 1]),
            sample_index=tf.reshape(element.get('sample_index', 0), [-1, 1]),
            client_id=tf.reshape(element.get('client_id', 'unknown'), [-1, 1]),
            filename=tf.reshape(element.get('filename', ''), [-1, 1])
        )

    dataset = dataset.map(convert_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        test_num_data = test_num_data + 1
        base_filename = f"/code/data/augmented/client_{client_id}/batch_{test_num_data:04d}"
        Path(f"/code/data/augmented/client_{client_id}").mkdir(parents=True, exist_ok=True)
        
        # Χρήση της απλοποιημένης έκδοσης
        dataset = apply_augmentation_pipeline_simple(dataset, base_filename, client_id, num_augm=NUM_AUGMENTATION)
    
    return dataset.shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids, is_training):
  return [
      preprocess(client_data[x], x, is_training)

      for x in client_ids
  ]


def preprocess_testing_dataset(dataset):

    def batch_format_fn(element):
        return OrderedDict(
            paired_input_data=tf.reshape(element['pixels'], [-1, 224, 224, 1]),
            paired_input_labels=tf.reshape(element['label'], [-1, 1])
        )
    return dataset.batch(1).map(batch_format_fn).prefetch(3)



data = Dataset()
training_dataset, evaluation_dataset, test_dataset = data.load_dataset()
sample_clients = list(training_dataset.keys())[:]
federated_train_data = make_federated_data(training_dataset, sample_clients, is_training=True)





def save_federated_augmented_images(federated_data, base_save_path, client_ids):
    """
    Αποθηκεύει τις augmented εικόνες από το federated dataset.
    Οργανώνει τις εικόνες ανά client και κρατάει στατιστικά.
    
    Args:
        federated_data: List of tf.data.Dataset (preprocessed με augmentation)
        base_save_path: Base directory για αποθήκευση
        client_ids: List με τα client IDs
    """
    
    base_path = Path(base_save_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Στατιστικά για όλους τους clients
    total_stats = {
        'total_images': 0,
        'original_count': 0,
        'forgery_count': 0,
        'clients_processed': 0
    }
    
    print("=" * 80)
    print("ΑΠΟΘΗΚΕΥΣΗ FEDERATED AUGMENTED IMAGES")
    print("=" * 80)
    
    for client_idx, (client_dataset, client_id) in enumerate(zip(federated_data, client_ids)):
        print(f"\n📁 Client: {client_id} [{client_idx + 1}/{len(client_ids)}]")
        
        # Δημιουργία φακέλου για τον client
        client_folder = base_path / f"client_{client_id}"
        client_folder.mkdir(parents=True, exist_ok=True)
        
        # Counters για τον client
        client_stats = defaultdict(int)
        image_counter = 0
        
        # Iterate στα batches
        for batch_idx, batch in enumerate(client_dataset):
            x_batch = batch['x']  # Shape: [batch_size, 224, 224, 3]
            y_batch = batch['y']  # Shape: [batch_size, 1]
            
            # Iterate στις εικόνες του batch
            for img_idx in range(x_batch.shape[0]):
                image_array = x_batch[img_idx].numpy()
                label = int(y_batch[img_idx].numpy()[0])
                
                # Normalize pixel values αν χρειάζεται
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
                
                # Clip values για safety
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                
                # Δημιουργία PIL Image
                pil_image = Image.fromarray(image_array, mode='RGB')
                
                # Label name
                label_name = "original" if label == 1 else "forgery"
                
                # Filename με counter
                filename = f"{label_name}_{image_counter:05d}.png"
                save_path = client_folder / filename
                
                # Αποθήκευση
                pil_image.save(save_path, format='PNG')
                
                # Update stats
                client_stats[label_name] += 1
                image_counter += 1
                
                # Progress indicator
                if image_counter % 50 == 0:
                    print(f"  ✓ Saved {image_counter} images...", end='\r')
        
        # Client summary
        print(f"  ✓ Saved {image_counter} images (originals: {client_stats['original']}, "
              f"forgeries: {client_stats['forgery']})")
        
        # Update total stats
        total_stats['total_images'] += image_counter
        total_stats['original_count'] += client_stats['original']
        total_stats['forgery_count'] += client_stats['forgery']
        total_stats['clients_processed'] += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total clients processed: {total_stats['clients_processed']}")
    print(f"Total images saved: {total_stats['total_images']}")
    print(f"  - Original images: {total_stats['original_count']}")
    print(f"  - Forgery images: {total_stats['forgery_count']}")
    print(f"Save location: {base_path.absolute()}")
    print("=" * 80)
    
    return total_stats


def save_with_batch_info(federated_data, base_save_path, client_ids):
    """
    Εναλλακτική έκδοση που κρατάει πληροφορία batch στο filename.
    Χρήσιμο για debugging.
    """
    
    base_path = Path(base_save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int)) # . . .
    
    for client_dataset, client_id in zip(federated_data, client_ids):
        print(f"\nProcessing client: {client_id}")
        
        client_folder = base_path / f"client_{client_id}"
        client_folder.mkdir(parents=True, exist_ok=True)
        
        total_images = 0
        batch_counter=0
        
        for batch_idx, batch in enumerate(client_dataset):
            images = batch['x'].numpy()
            labels = batch['y'].numpy()

            has_metadata = 'is_original' in batch
            
            for img_idx in range(images.shape[0]):
                img = images[img_idx]
                label = int(labels[img_idx][0])
                
                # Normalize pixel values
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Δημιουργία PIL Image
                pil_image = Image.fromarray(img, mode='RGB')
                
                # Προσδιορισμός ονόματος βάσει metadata
                if has_metadata:
                    is_original = bool(batch['is_original'][img_idx][0])
                    aug_index = int(batch['aug_index'][img_idx][0])
                    sample_index = int(batch['sample_index'][img_idx][0])
                    
                    if is_original:
                        aug_type = "original"
                        iteration = 0
                    else:
                        aug_type = f"aug_{aug_index:02d}"
                        iteration = aug_index + 1
                    
                    filename = (
                        f"client_{client_id}_"
                        f"sample{sample_index:04d}_"
                        f"{aug_type}_"
                        f"iter{iteration:02d}_"
                        f"{'original' if label == 1 else 'forgery'}_"
                        f"batch{batch_idx:04d}_"
                        f"img{img_idx:02d}.png"
                    )
                    
                    # Update stats
                    if is_original:
                        stats[client_id]['original_images'] += 1
                    else:
                        stats[client_id]['augmented_images'] += 1
                        stats[client_id][f'aug_iter_{iteration}'] += 1
                else:
                    # Fallback αν δεν υπάρχουν metadata
                    filename = (
                        f"client_{client_id}_"
                        f"batch{batch_idx:04d}_"
                        f"img{img_idx:02d}_"
                        f"{'original' if label == 1 else 'forgery'}.png"
                    )
                
                # Αποθήκευση
                save_path = client_folder / filename
                pil_image.save(save_path, format='PNG')
                
                batch_counter += 1
                if batch_counter % 20 == 0:
                    print(f"  ✓ Saved {batch_counter} images...", end='\r')


# Για δομή με batch info (για debugging):
save_path_debug = "/code/data/federated_images_with_batches"
save_with_batch_info(federated_train_data, save_path_debug, sample_clients)

'''