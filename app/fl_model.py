from modifier.Client_dataset import Dataset
from version_test_runtime import print_usage_versions_and_configuration_cpu_gpu_usage
import utils 
import models as model
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import keras
import tensorflow_addons as tfa
import tensorflow_federated as tff

import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns
import os
import random
from collections import OrderedDict, defaultdict
import collections

from load_tff_dataset import Load_tff_dataset 





tf.keras.backend.clear_session()

print_usage_versions_and_configuration_cpu_gpu_usage()
precision = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(precision)

# seed = 25 
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# Πιο αυστηρό seeding
tf.config.experimental.enable_op_determinism()


def apply_augmentation_pipeline(dataset, num_augm=1):
    num_augmentations = num_augm  

    def augment_func(sample, seed_aug=42):
        pixels = sample['pixels']
        label = sample['label']
              
        pixels = tf.cast(pixels, tf.float32)
        # if len(pixels.shape) == 2:
        #     pixels = tf.expand_dims(pixels, axis=-1)
        
        pixels = tf.ensure_shape(pixels, [240, 240, 3])
        pixels = tf.image.stateless_random_contrast(pixels, 0.9, 1.2, seed=(seed_aug+2000,0))
        angle = tf.random.stateless_uniform([], seed=(seed_aug+100, 0), minval=-0.2, maxval=0.2)
        pixels = tfa.image.rotate(
                pixels,
                angle,
                interpolation='NEAREST',
                fill_mode='constant',
                fill_value=1.0 
            )   
        
        return OrderedDict(pixels=pixels, label=label)
    
    # Create dataset with original samples
    augmented_original_dataset = dataset
    
    for i in range(0, num_augmentations):
        augmented_dataset = dataset.map(
            lambda x: augment_func(x, seed_aug=42 + i), num_parallel_calls=tf.data.AUTOTUNE)
        
        augmented_original_dataset = augmented_original_dataset.concatenate(augmented_dataset)
        print(f"The augmented Images are: {i}")

    return augmented_original_dataset


NUM_CLIENTS = 30
NUM_EPOCHS = 2
BATCH_SIZE = 2
SHUFFLE_BUFFER = 50 # 50
PREFETCH_BUFFER = tf.data.AUTOTUNE # 776
NUM_ROUNDS = 400


def preprocess(dataset, is_training=False):

    def convert_to_rgb(element):
        pixels = element['pixels']
        label = element['label']

        pixels = tf.image.grayscale_to_rgb(pixels)
        pixels = tf.ensure_shape(pixels, [240, 240, 3])

        
        return OrderedDict(pixels=pixels, label=label)
    
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
                x=tf.reshape(element['pixels'], [-1, 240, 240, 3]),
                y=tf.reshape(element['label'], [-1, 1])
            )


    dataset = dataset.map(convert_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = apply_augmentation_pipeline(dataset, num_augm=1)  

    
    
    # return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
    #         BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

    return dataset.shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids, is_training):
  return [
      preprocess(client_data[x], is_training)

      for x in client_ids
  ]


input_shape = (240, 240, 3)
def create_model(): 
    # simple_cnn_model = model.create_model_base_simple_cnn_model(input_shape)
    # cnn_with_many_filters = create_model_cnn_with_many_filters(input_shape)
    # cnn_with-many_blocks = create_model_cnn_with_many_blocks(input_shape)
    denseNet_transfer_learning_model = model.create_model_transfer_learning_DesNet121(input_shape)

    return denseNet_transfer_learning_model


def model_fn():
  keras_model = create_model()
  keras_model.summary()

  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec= OrderedDict([
      ('x', tf.TensorSpec(shape=(None, 240, 240, 3), dtype=tf.float32)),  # Input shape
      ('y', tf.TensorSpec(shape=(None, 1), dtype=tf.int32))  # Output shape
  ]),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(
                        name='auc',
                        curve='ROC',
                        summation_method="interpolation",
                        num_thresholds=200
                    ),
        tf.keras.metrics.FalseNegatives(name='false_negatives'),
        tf.keras.metrics.FalsePositives(name='false_positives'),
        tf.keras.metrics.TrueNegatives(name='true_negatives'),
        tf.keras.metrics.TruePositives(name='true_positives')
        ])



data = Dataset()
training_dataset, evaluation_dataset, test_dataset = data.load_dataset()
sample_clients = list(training_dataset.keys())[:NUM_CLIENTS]
federated_train_data = make_federated_data(training_dataset, sample_clients, is_training=True)

shuffled_eval_clients = list(evaluation_dataset.keys())
shuffled_eval_clients.sort()  
sample_eval_clients = shuffled_eval_clients[:NUM_CLIENTS]
federated_test_data = make_federated_data(evaluation_dataset, sample_eval_clients, is_training=False)

print("evaluation dataset:")
print(evaluation_dataset)
print("")
print("training dataset")
print(training_dataset)
print("")


        

def plot_metrics(train_metrics, eval_metrics, metric_name, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(eval_metrics, label=f'Evaluation {metric_name}')
    plt.xlabel('Rounds')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} over Training Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

    filename = f"/code/modifier/plots/{metric_name.lower().replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()  # Clear the figure to free memory

# =================================================================================
def save_lr_in_csv(client_lr, server_lr, data, best_auc, num):
    csv_file=f"/code/modifier/lr_auc_results/learning_rates_{num}.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["client_lr       ", "server_lr      ", "best_eval_loss     ",  "best_uac    "])

        # Append data rows
        # for clr, slr in zip(client_lr, server_lr):
        writer.writerow([client_lr, "       ", server_lr, "       ", data*100, "       ", best_auc*100])

    print(f"Data successfully appended to {csv_file}.")


server_learning_rates = np.logspace(-3, 1, num=9) 
client_learning_rates = np.logspace(-3, 1, num=9)
data_rows = []

for server_lr in reversed(server_learning_rates):
    data_row = []

    for client_lr in client_learning_rates:

        training_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn,
            client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=np.float32(client_lr)),
            server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=np.float32(server_lr))
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
        patience = 6 # Number of rounds to wait before stopping
        restore_best_weights = True
        best_model_weights = None

        for round_num in range(1, NUM_ROUNDS):
            result = training_process.next(train_state, federated_train_data)
            train_state = result.state
            train_metrics = result.metrics['client_work']['train']
            print('round {:2d}, metrics={}'.format(round_num, train_metrics))

            # Store the training metrics for this round
            train_metrics = result.metrics['client_work']['train']
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


            # # Early stopping logic
            #-----------------------------
            if current_auc > best_auc:
            #     best_auc = current_auc
            # if current_eval_loss  < best_eval_loss:
                best_eval_loss = current_eval_loss
                best_auc = current_auc
                patience_counter = 0
                if restore_best_weights:
                    best_model_weights = model_weights
            else:
                patience_counter += 1
                print(f"No AUC improvement for {patience_counter} rounds. Best AUC: {best_auc:.4f}, and loss: {best_eval_loss}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered at round {round_num}!")
                    if restore_best_weights and best_model_weights is not None:
                        print("Restoring best model weights")
                        train_state = training_process.set_model_weights(train_state, best_model_weights)
                    break


        data_row.append(best_auc)


        # Plot all metrics
        plot_metrics(train_metrics_history['loss'], 
                    eval_metrics_history['eval_loss'], 
                    'Loss', 'Loss Value')

        plot_metrics(train_metrics_history['binary_accuracy'], 
                    eval_metrics_history['binary_accuracy'], 
                    'Binary Accuracy', 'Accuracy')

        plot_metrics(train_metrics_history['auc'], 
                    eval_metrics_history['auc'], 
                    'AUC', 'AUC Value')
        
        plot_metrics(train_metrics_history['false_negatives'], 
                    eval_metrics_history['false_negatives'], 
                    'false Negatives_tr', 'false_positives_ev')
        

        save_lr_in_csv(client_lr, server_lr, best_eval_loss, best_auc, 9) 
    
    print("==========================================================")
    print("")
    # print(f"client_lr: {client_lr} , server_lr: {server_lr} , best_auc: {best_auc}")    
    print(f"client_lr: {client_lr} , server_lr: {server_lr} , best_eval_loss: {best_eval_loss}")      
    data_rows.append(data_row)
    print("==========================================================")



data = np.array(data_rows)

plt.figure(figsize=(20, 18))
sns.heatmap(
    data, annot=True, fmt=".2f", cmap="YlGn", cbar_kws={'label': 'Accuracy'},
    # xticklabels=[f"{x:.6f}" for x in client_learning_rates], yticklabels=[f"{x:.6f}" for x in server_learning_rates]
    xticklabels=[f"{x}" for x in client_learning_rates], yticklabels=[f"{x}" for x in server_learning_rates]
)

plt.title("Grid Search - Learning Rate")
plt.xlabel("Client Learning Rates")
plt.ylabel("Server Learning Rates")
plt.tight_layout()
plt.show()

filename = f"/code/modifier/plots/grid_search_plot_{len(server_learning_rates)}.png"
plt.savefig(filename)
print(f"Saved plot to {filename}")
plt.close()  # Clear the figure to free memory
print(train_metrics_history['false_negatives'])
print(eval_metrics_history['false_negatives'])