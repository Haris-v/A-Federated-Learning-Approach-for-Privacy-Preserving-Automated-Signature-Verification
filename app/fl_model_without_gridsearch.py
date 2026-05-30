from Client_dataset import Dataset
from version_test_runtime import print_usage_versions_and_configuration_cpu_gpu_usage
import models as model
import utils 
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, mixed_precision
from tensorflow_federated.python.learning.optimizers.keras_optimizer import KerasOptimizer
import keras
import sys
import typing

# Monkey patch για το typing issue
if sys.version_info < (3, 9):
    import typing_extensions
    sys.modules['typing'] = typing_extensions
import tensorflow_federated as tff
import nest_asyncio
nest_asyncio.apply()
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns
import os
import random
from collections import OrderedDict, defaultdict
import collections
from pathlib import Path
import gc


from load_tff_dataset import Load_tff_dataset 



tf.keras.backend.clear_session()

print_usage_versions_and_configuration_cpu_gpu_usage()
precision = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(precision)

# seed = 25 
seed = 42
test_num_data = 0

random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

tf.config.experimental.enable_op_determinism()

def apply_augmentation_pipeline(dataset, base_filename, seed_list_client_selector, num_augm=1):
    num_augmentations = num_augm  
    current_num_augm_iter=0

    def augment_func(sample, base_filename, sample_index, aug_index, current_num_augm_iter,seed_aug=42):
        pixels = sample['pixels']
        label = sample['label']

        pixels = tf.cast(pixels, tf.float32)
        # if len(pixels.shape) == 2:
        #     pixels = tf.expand_dims(pixels, axis=-1)

        pixels = tf.ensure_shape(pixels, [224, 224, 3])
        
        angle = tf.random.stateless_uniform(
            [], 
            seed=(seed_aug, current_num_augm_iter), 
            minval= -0.7, 
            maxval= 0.7 
            )

        
        rotated_pixels = tfa.image.rotate(
                pixels,
                angle,
                interpolation='NEAREST',
                fill_mode='constant',
                fill_value=1.0 
            )   

        
        pixels_contrast = tf.image.stateless_random_contrast(rotated_pixels, 
                                                    0.7, 
                                                    1.4, 
                                                    seed=(seed_aug,current_num_augm_iter)
                                                    )
        
        pixels_contrast = tf.clip_by_value(pixels_contrast, 0.0, 1.0)
        pixels_contrast = pixels_contrast * 255
        aug_index_str = tf.as_string(aug_index)

        # filename = f"{base_filename}_sample{sample_index_str}_aug{aug_index_str}.png"
        filename = f"{base_filename}_aug{aug_index_str}.png"
        pixels_save_image = tf.cast(pixels_contrast, tf.uint8)
        image_encoded = tf.io.encode_png(pixels_save_image)
        tf.io.write_file(filename, image_encoded)

        return OrderedDict(pixels=pixels_contrast, label=label)

    def create_brightness_sample(sample, base_filename, sample_index, aug_index, current_num_augm_iter, seed_aug=42):
        pixels = sample['pixels']
        label = sample['label']
        pixels = tf.cast(pixels, tf.float32)
        pixels = tf.ensure_shape(pixels, [224, 224, 3])
        
        brightness_pixels = tf.image.stateless_random_brightness(
            pixels,
            max_delta=0.15,
            seed=(seed_aug + 100, current_num_augm_iter) 
        )
        
        brightness_pixels = tf.clip_by_value(brightness_pixels, 0.0, 1.0)
        brightness_pixels = brightness_pixels * 255

        sample_index_str = tf.as_string(sample_index)
        aug_index_str = tf.as_string(aug_index)

        # filename = f"{base_filename}_sample{sample_index_str}_aug{aug_index_str}.png"
        filename = f"{base_filename}_aug{aug_index_str}.png"
        pixels_save_image = tf.cast(brightness_pixels, tf.uint8)
        image_encoded = tf.io.encode_png(pixels_save_image)
        tf.io.write_file(filename, image_encoded)
        
        return OrderedDict(pixels=brightness_pixels, label=label)
    
    def create_noisy_samples(sample, base_filename, sample_index, aug_index, current_num_augm_iter, seed_aug=42):
        pixels = sample['pixels']
        label = sample['label']
        pixels = tf.cast(pixels, tf.float32)
        pixels = tf.ensure_shape(pixels, [224, 224, 3])
    
        noise = tf.random.stateless_normal(
            shape=tf.shape(pixels),
            seed=(seed_aug + 200, current_num_augm_iter),
            stddev=0.015  
        )
        pixels_noisy = pixels + noise
        pixels_final = tf.clip_by_value(pixels_noisy, 0.0, 1.0)
        pixels_final = pixels_final * 255

        aug_index_str = tf.as_string(aug_index)
        filename = f"{base_filename}_aug{aug_index_str}.png"
        pixels_save_image = tf.cast(pixels_final, tf.uint8)
        image_encoded = tf.io.encode_png(pixels_save_image)
        tf.io.write_file(filename, image_encoded)
        return OrderedDict(pixels=pixels_final, label=label)
    
    def normalize_original(sample):
        pixels = tf.cast(sample['pixels'], tf.float32) * 255.0
        return OrderedDict(pixels=pixels, label=sample['label'])
    
    original_dataset = dataset.map(normalize_original, num_parallel_calls=tf.data.AUTOTUNE)
    # # Create dataset with original samples
    all_datasets = [original_dataset]

    for i in range(0, num_augmentations):

        current_num_augm_iter = current_num_augm_iter + i
        augmented_dataset = dataset.enumerate().map(
            lambda idx, x: augment_func(x, base_filename, idx, i, current_num_augm_iter, seed_aug=42 + i+ seed_list_client_selector), 
            num_parallel_calls=tf.data.AUTOTUNE
        )

        len_dataset = len(dataset)
        decision_seed = (42 + i , current_num_augm_iter)
        prob = tf.random.stateless_uniform([], seed=decision_seed, minval=0.0, maxval=1.0)
        tf.print("prob:")
        tf.print(prob)
        tf.print(f"Tf dataset len: {len(dataset)}")
        
        # if prob > 0.5:
        #     tf.print("")
        #     tf.print("Brightness!!!!")
        #     tf.print("prob:")
        #     tf.print(prob)
        #     tf.print("")
        #     brightness_dataset = dataset.enumerate().map(
        #         lambda idx, x: create_brightness_sample(x, base_filename, idx, i, current_num_augm_iter, seed_aug=42 + i), 
        #         num_parallel_calls=tf.data.AUTOTUNE
        #     )
        #     all_datasets.append(brightness_dataset)

        all_datasets.append(augmented_dataset)
    for i in range(1):
        brightness_dataset = dataset.enumerate().map(
                    lambda idx, x: create_brightness_sample(x, base_filename, idx, i, current_num_augm_iter, seed_aug=42 + i+seed_list_client_selector), 
                    num_parallel_calls=tf.data.AUTOTUNE
                )
        all_datasets.append(brightness_dataset)

    final_dataset = all_datasets[0]
    for aug_dataset in all_datasets[1:]:
        final_dataset = final_dataset.concatenate(aug_dataset)
    
    tf.print("")
    tf.print("")
    tf.print("THE SIZE OF AUGMENTED AND ORIGINAL DATASET")
    tf.print("")
    tf.print("")
    tf.print(len(all_datasets))
    tf.print("")
    tf.print("")
    tf.print("---------------------------------------------------------------------------")

    total_samples = 0
    for i, ds in enumerate(all_datasets):
        count = sum(1 for _ in ds)
        total_samples += count
        tf.print(f"Dataset {i}: {count} samples")
    
    tf.print(f"Total samples in final dataset: {total_samples}")
    tf.print("---------------------------------------------------------------------------")
    return final_dataset



NUM_CLIENTS = 30
NUM_EPOCHS = 5# 8 #0
BATCH_SIZE = 9#10 #7
SHUFFLE_BUFFER = 1400
PREFETCH_BUFFER = tf.data.AUTOTUNE
NUM_ROUNDS = 101
NUM_AUGMENTATION=1

test_num_data = 0
def preprocess(dataset, client_id, seed_list_client_selector, is_training=False):
    global test_num_data

    def convert_to_rgb(element):
        pixels = element['pixels']
        label = element['label']

        pixels = tf.image.grayscale_to_rgb(pixels)
        pixels = tf.ensure_shape(pixels, [224, 224, 3])

        
        return OrderedDict(pixels=pixels, label=label)
    
    def normalize_original(sample):
        pixels = tf.cast(sample['pixels'], tf.float32) * 255.0
        return OrderedDict(pixels=pixels, label=sample['label'])
    
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
                x=tf.reshape(element['pixels'], [-1, 224, 224, 3]),
                y=tf.reshape(element['label'], [-1, 1])
            )


    dataset = dataset.map(convert_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    

    if is_training:
        test_num_data = test_num_data + 1
        base_filename = f"/code/data/augmented/client_{client_id}/batch_{test_num_data}"
        Path(f"/code/data/augmented/client_{client_id}").mkdir(parents=True, exist_ok=True)
        dataset = apply_augmentation_pipeline(dataset, base_filename, seed_list_client_selector, num_augm=NUM_AUGMENTATION)  
    else:
        dataset = dataset.map(normalize_original, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
            BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

    # return dataset.shuffle(SHUFFLE_BUFFER, seed=1).batch(
    #   BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids,seed_list_client_selector, is_training):
  return [
      preprocess(client_data[x], x, seed_list_client_selector, is_training)

      for x in client_ids
  ]


def preprocess_testing_dataset(dataset):

    
    def convert_to_rgb(element):
        pixels = element['pixels']
        label = element['label']

        pixels = tf.image.grayscale_to_rgb(pixels)
        pixels = tf.ensure_shape(pixels, [224, 224, 3])
        
        return OrderedDict(pixels=pixels, label=label)
    
    def normalize_original(sample):
        pixels = tf.cast(sample['pixels'], tf.float32) * 255.0
        return OrderedDict(pixels=pixels, label=sample['label'])

    def batch_format_fn(element):
        return OrderedDict(
            paired_input_data=tf.reshape(element['pixels'], [-1, 224, 224, 3]),
            paired_input_labels=tf.reshape(element['label'], [-1, 1])
        )
    
    dataset = dataset.map(normalize_original, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(convert_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(1).map(batch_format_fn).prefetch(3)


def make_testing_federated_data(client_data, client_ids):
  return [
      preprocess_testing_dataset(client_data[x])

      for x in client_ids
  ]


input_shape = (224, 224, 3)
def create_model(): 
    # simple_cnn_model = model.create_model_base_simple_cnn_model(input_shape)
    # cnn_with_many_filters = create_model_cnn_with_many_filters(input_shape)
    # cnn_with-many_blocks = create_model_cnn_with_many_blocks(input_shape)
    denseNet_transfer_learning_model = model.create_model_transfer_learning_DesNet121(input_shape)
    # resNet50V2_transfer_learning_model = model.create_model_transfer_learning_ResNet50V2(input_shape)
    # efficientNetV2B0 = model.create_model_transfer_learning_EfficientNetV2B0(input_shape)
    # mobileNetV3 = model.create_model_transfer_lerning_mobileNetV3Small(input_shape)
    return denseNet_transfer_learning_model
    # return efficientNetV2B0
    # return mobileNetV3

    

thresholds = list(np.linspace(0, 1, 100))
def model_fn():
  keras_model = create_model()
  keras_model.summary()

  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec= OrderedDict([
      ('x', tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)),  # Input shape
      ('y', tf.TensorSpec(shape=(None, 1), dtype=tf.int32))  # Output shape
  ]),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.AUC(
                        name='auc',
                        curve='ROC',
                        summation_method="interpolation",
                        num_thresholds=200
                    ),
        tf.keras.metrics.FalseNegatives(name='false_negatives', thresholds=thresholds),
        tf.keras.metrics.FalsePositives(name='false_positives', thresholds=thresholds),
        tf.keras.metrics.TrueNegatives(name='true_negatives', thresholds=thresholds),
        tf.keras.metrics.TruePositives(name='true_positives', thresholds=thresholds),
        # tf.keras.metrics.FalseNegatives(name='false_negatives', thresholds=0.0202),
        # tf.keras.metrics.FalsePositives(name='false_positives', thresholds=0.0202),
        # tf.keras.metrics.TrueNegatives(name='true_negatives', thresholds=0.0202),
        # tf.keras.metrics.TruePositives(name='true_positives', thresholds=0.0202)
        ])


def model_fn_pred(model_weights, test_data, eer_threshold, model_save_dir_path, save_model_flag):
        """
        Label 0 = Forgery
        Label 1 = Genuine
        Prediction < threshold = Forgery
        Prediction > threshold = Genuine

        
        """

        dl_model = create_model()
        dl_model.summary()
        dl_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00031622776601683794),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                    tf.keras.metrics.AUC(
                        name='auc',
                        curve='ROC',
                        summation_method="interpolation",
                        num_thresholds=200
                    )
                    # tf.keras.metrics.FalseNegatives(name='false_negatives', thresholds=eer_threshold),
                    # tf.keras.metrics.FalsePositives(name='false_positives', thresholds=eer_threshold),
                    # tf.keras.metrics.TrueNegatives(name='true_negatives', thresholds=eer_threshold),
                    # tf.keras.metrics.TruePositives(name='true_positives', thresholds=eer_threshold)
                ])
        
        model_weights.assign_weights_to(dl_model)

        if save_model_flag:
            utils.save_model(model_save_dir_path,dl_model)
        
        # 2. Process each client's dataset separately
        truePositives = 0  #TP: Genuine correctly classified as Genuine
        falseNegatives = 0 #FN: Genuine incorrectly classified as Forgry
        trueNegatives = 0  #TN: Forgery correctly classified as Forgery
        falsePositives = 0 #FP: Forgery incorrectly classified as Genuine 
        all_predictions = []
        for client_idx, client_dataset in enumerate(test_data):
            print(f"\n Processing Client {client_idx + 1}")
            iteration_counter_per_client=0

            # Convert dataset to numpy batches
            client_size = len(client_dataset)
            for batch in client_dataset:

                if isinstance(batch, dict):
                    inputs = batch['paired_input_data']  
                    labels = batch["paired_input_labels"]
                else:
                    inputs = batch
                
                # Make predictions
                batch_predictions = dl_model.predict(inputs)


                if labels == 1: 
                    if batch_predictions >= eer_threshold:
                          truePositives += 1
                    else:
                          falseNegatives += 1 # forgery_inaccurate_counter
                else:
                    if batch_predictions < eer_threshold:
                          trueNegatives += 1
                    else:
                          falsePositives += 1 # genuine_inaccurate_counter

                all_predictions.append(batch_predictions)
                iteration_counter_per_client += 1
                
                print("======================================================")
                print("")
                print(f"iteration step: {iteration_counter_per_client} , cliend dataset size: {client_size}")
                tf.print(f"Label of prediction {labels}  and the shape is {type(labels)}")
                # print(batch_predictions)
            print("=============New Client====================================")
            print(f"Total Predictions for this client: {iteration_counter_per_client} with threshold {eer_threshold}")
            print("===================================================================")
            print("==================================================================")

        print("Return!") 
        return all_predictions, truePositives, falseNegatives, trueNegatives, falsePositives


data = Dataset()
training_dataset, evaluation_dataset, test_dataset, test_dataset_without_sf = data.load_dataset()

shuffled_eval_clients = list(evaluation_dataset.keys())
shuffled_eval_clients.sort()  
sample_eval_clients = shuffled_eval_clients[:NUM_CLIENTS]
federated_test_data = make_federated_data(evaluation_dataset, sample_eval_clients, None, is_training=False)



training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,                                                               
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.0001), 
        server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.00031622776601683794) 
            )

# Construct federated evaluation computation here:
evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn)
train_state = training_process.initialize()
evaluation_state = evaluation_process.initialize()
        

    
# Initialize variables for metrics history
train_metrics_history = defaultdict(list)
eval_metrics_history = defaultdict(list)

# Initialize variables for early stopping
best_auc = 0.0
best_eval_loss = 1.0
current_auc = 0
patience_counter = 0
patience =  8 #12 # 10 with the best auc result 90%+ Number of rounds to wait before stopping
restore_best_weights = True
best_model_weights = None
number_of_client_selection= 5
seed_list_client_selector = list(range(0, 401))
for round_num in range(0, NUM_ROUNDS):
    # number_of_client_selection = random.randint(2, 7) # next 2-5
    select_seed_per_round = seed_list_client_selector[round_num]
    clients_keys = list(training_dataset.keys())
    clients_selected_keys = list(utils.select_idx_for_clients_participation(number_of_client_selection, select_seed_per_round, clients_keys))
    federated_train_data = make_federated_data(training_dataset, clients_selected_keys, seed_list_client_selector[round_num], is_training=True)

    result = training_process.next(train_state, federated_train_data)
    train_state = result.state

    train_metrics = result.metrics['client_work']['train']
    print('round {:2d}, metrics={}'.format(round_num, train_metrics))

    # Store the training metrics for this round
    train_metrics_history['loss'].append(train_metrics['loss'])
    train_metrics_history['binary_accuracy'].append(train_metrics['binary_accuracy'])
    train_metrics_history['auc'].append(train_metrics['auc'])
    train_metrics_history['false_negatives'].append(train_metrics['false_negatives'])
    train_metrics_history['false_positives'].append(train_metrics['false_positives'])

    # ------Evaluation Process-------

    # Extract the trained model weights
    model_weights = training_process.get_model_weights(train_state)
    # Set the trained model weights in the evaluation state
    evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
    # Run the evaluation process
    evaluation_result = evaluation_process.next(evaluation_state, federated_test_data)
    evaluation_metrics = evaluation_result.metrics['client_work']['eval']
    print('round {:2d}, Evaluation metrics={}'.format(round_num, evaluation_metrics))

    eval_loss=evaluation_metrics['current_round_metrics']['loss']
    current_eval_loss = eval_loss
    eval_metrics_history['eval_loss'].append(eval_loss)
    eval_binary_accuracy=evaluation_metrics['current_round_metrics']['binary_accuracy']
    eval_metrics_history['binary_accuracy'].append(eval_binary_accuracy)
    eval_auc=evaluation_metrics['current_round_metrics']['auc']
    eval_metrics_history['auc'].append(eval_auc) 
    current_auc = eval_auc
    eval_false_negatives = evaluation_metrics['current_round_metrics']['false_negatives']
    eval_metrics_history['false_negatives'].append(eval_false_negatives)
    eval_false_positives = evaluation_metrics['current_round_metrics']['false_positives']
    eval_metrics_history['false_positives'].append(eval_false_positives)
    eval_true_negatives = evaluation_metrics['current_round_metrics']['true_negatives']
    eval_metrics_history['true_negatives'].append(eval_true_negatives)
    eval_true_positives = evaluation_metrics['current_round_metrics']['true_positives']
    eval_metrics_history['true_positives'].append(eval_true_positives)

    frr, far = utils.calculate_frr_far_metrics(eval_false_negatives, eval_false_positives, eval_true_negatives, eval_true_positives)
    
    gc.collect()
    # # Early stopping logic
    #-----------------------------
    if current_auc > best_auc:
        best_eval_loss = current_eval_loss
        best_auc = current_auc
        patience_counter = 0
        if restore_best_weights:
            best_model_weights = model_weights
    else:
        patience_counter += 1
        print(f"No AUC improvement for {patience_counter} rounds. Best AUC: {best_auc:.4f}, and loss: {best_eval_loss:.4f}")
        if patience_counter >= patience:
            print(f"Early stopping triggered at round {round_num}!")
            if restore_best_weights and best_model_weights is not None:
                print("Restoring best model weights")
                train_state = training_process.set_model_weights(train_state, best_model_weights)
            break
    


# Comment for testing model
eer, eer_threshold, far_history, frr_history = utils.calculate_eer(eval_metrics_history, thresholds)

print("")
print("")
print("===========================================================================================")
print("")
print("-----The EER and EER threshold are:")
print(f"EER: {eer:.4f}, EER Threshold: {eer_threshold:.4f}")
print("")
print(f"Batch size: {BATCH_SIZE}, Local num epochs: {NUM_EPOCHS}")
print("")
print("===========================================================================================")
print("")
print("")


utils.create_plots_metrics(train_metrics_history, eval_metrics_history)
utils.plot_metrics_EER_threshold(thresholds, far_history, frr_history, eer_threshold, eer)


# ================Prediction=================================================

test_data = test_dataset[0]
test_labels = test_dataset[1]
try:
    test_dataset_without_sf_data = test_dataset_without_sf[0]
    test_labels_dataset_without_sf_data = test_dataset_without_sf[1]
except:
    test_dataset_without_sf_data = test_dataset_without_sf[0]
    print("Except")

sample_clients = list(test_data.keys())[:]
federated_testing_data = make_testing_federated_data(test_data, sample_clients)

sample_clients_without_sf = list(test_dataset_without_sf_data.keys())[:]
federated_test_data_without_sf = make_testing_federated_data(test_dataset_without_sf_data, sample_clients_without_sf)

trained_weights = train_state.global_model_weights

model_save_dir_path=f"/code/modifier/trained_model_weights"

# Test model with skilled forgeries
predictions, truePositives, falseNegatives, trueNegatives, falsePositives = model_fn_pred(trained_weights, 
                                                                                          federated_testing_data, 
                                                                                          eer_threshold, 
                                                                                          model_save_dir_path, 
                                                                                          True)
accuracy, precision, recall, f1_score = utils.calculate_metrics(len(predictions), 
                                                           truePositives, 
                                                           falseNegatives, 
                                                           trueNegatives, 
                                                           falsePositives)

# Test model without the skilled forgeries
predictions_without_sf, truePositives_without_sf, falseNegatives_without_sf, trueNegatives_without_sf, falsePositives_without_sf = model_fn_pred(trained_weights, 
                                                                                          federated_test_data_without_sf, 
                                                                                          eer_threshold, 
                                                                                          model_save_dir_path, 
                                                                                          False)


accuracy_without_sf, precision_without_sf, recall_without_sf, f1_score_without_sf = utils.calculate_metrics(len(predictions_without_sf), 
                                                           truePositives_without_sf, 
                                                           falseNegatives_without_sf, 
                                                           trueNegatives_without_sf, 
                                                           falsePositives_without_sf)
print(f"Len of predictions: {len(predictions)}")  

print("")
print(f"truePositives: {truePositives}")
print(f"falseNegatives: {falseNegatives}")
print(f"trueNegatives: {trueNegatives}")
print(f"falsePositives: {falsePositives}")
results_metadata = {
    "best_auc": "0.8605",
    "eval_loss": "0.5047",
    "final_round": 37,
    "num_epochs": 5,
    "batch_size": 9,
    "num_clients": 5,
    "eer": "0.2378",
    'eer_threshold': f"{eer_threshold:.4f}",
    'with_skilled_forgeries': {
        'predictions_count': len(predictions),
        'true_positives': int(truePositives),
        'false_negatives': int(falseNegatives),
        'true_negatives': int(trueNegatives),
        'false_positives': int(falsePositives),
        'accuracy': f"{accuracy:.4f}",
        'precision': f"{precision:.4f}",
        'recall': f"{recall:.4f}",
        'f1_score': f"{f1_score:.4f}"
    },
    'without_skilled_forgeries': {
        'predictions_count': len(predictions_without_sf),
        'true_positives': int(truePositives_without_sf),
        'false_negatives': int(falseNegatives_without_sf),
        'true_negatives': int(trueNegatives_without_sf),
        'false_positives': int(falsePositives_without_sf),
        'accuracy': f"{accuracy_without_sf:.4f}",
        'precision': f"{precision_without_sf:.4f}",
        'recall': f"{recall_without_sf:.4f}",
        'f1_score': f"{f1_score_without_sf:.4f}"
    }
}

utils.save_metadata(model_save_dir_path, metadata=results_metadata)