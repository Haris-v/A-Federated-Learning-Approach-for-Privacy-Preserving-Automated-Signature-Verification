import matplotlib.pyplot as plt
import random
import keras
import tensorflow as tf
import numpy as np
from pathlib import Path
import json




seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)



def select_idx_for_clients_participation_pair_idxs(NUM_CLIENTS):
    while True:
        idx1 = random.randint(0, NUM_CLIENTS)
        idx2 = random.randint(0, NUM_CLIENTS)
        diff = abs(idx1 - idx2)

        temp = idx1
        if idx1 > idx2:
            idx1 = idx2
            idx2 = temp
        
        if diff >= 16:  
            return idx1, idx2
        

def select_idx_for_clients_participation(number_of_client_selection, seed_client_selector, clients_keys):
    rng_for_client_Selector=random.Random(seed_client_selector)
    client_selector_list=[]
    selected_clients = set()
    while len(selected_clients) < number_of_client_selection:
        client_idx= rng_for_client_Selector.randint(0,len(clients_keys)-1)
        client = clients_keys[client_idx]
        client_selector_list.append(client)
        selected_clients = set(client_selector_list)
        
    tf.print("")
    tf.print("")
    tf.print("")
    tf.print("")
    tf.print("")
    tf.print("")
    tf.print("")
    tf.print("The selected clients are: ")
    tf.print(selected_clients)
    
    return selected_clients


def plot_metrics(train_metrics, eval_metrics, metric_name, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(eval_metrics, label=f' Evaluation {metric_name}')
    plt.xlabel('Rounds')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} over Training Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

    filename = f"/code/modifier/plots/{metric_name.lower().replace(' ', '_')}_plot.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()  


def plot_metrics_EER_threshold(thresholds, far_history, frr_history, eer_threshold, eer):
    # Comment for testing model
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_history, label=f'FAR', color='blue')
    plt.plot(thresholds, frr_history, label=f'FRR', color='red')
    # Mark EER point
    plt.scatter(eer_threshold, eer, color='black', s=100, label=f'EER = {eer:.3f}')
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=eer, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel("Error Rate")
    plt.title('FAR vs. FRR Curves (EER Highlighted)')
    plt.legend()
    plt.grid(True)
    plt.show()


    filename = f"/code/modifier/plots/FAR_FRR_EER_without_grid_search.png"

    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close() 

def calculate_frr_far_metrics(false_negatives, false_positives, true_negatives, true_positives):
    """
    Calculate False Rejection Rate (FRR) and False Acceptance Rate (FAR) from the provided metrics.
    
    Args:
        false_negatives (int): Number of false negatives.
        false_positives (int): Number of false positives.
        true_negatives (int): Number of true negatives.
        true_positives (int): Number of true positives.
    
    Returns:
        tuple: (FRR, FAR)
    """
    frr = false_negatives / (false_negatives + true_positives + tf.keras.backend.epsilon())  # False Rejection Rate
    far = false_positives / (false_positives + true_negatives  + tf.keras.backend.epsilon()) # False Acceptance Rate
    return frr, far



def calculate_eer(eval_metrics_history, thresholds):
    """Calculate EER using all threshold values from training history"""
    
    # Get the final metrics (last round)
    final_fn = eval_metrics_history['false_negatives'][-1]  
    final_fp = eval_metrics_history['false_positives'][-1] 
    final_tn = eval_metrics_history['true_negatives'][-1]   
    final_tp = eval_metrics_history['true_positives'][-1]   
    
    frr_history = []
    far_history = []
    
    # Calculate FRR and FAR for each threshold
    for i in range(len(thresholds)):
        fn = final_fn[i] if isinstance(final_fn, (list, np.ndarray)) else final_fn
        fp = final_fp[i] if isinstance(final_fp, (list, np.ndarray)) else final_fp
        tn = final_tn[i] if isinstance(final_tn, (list, np.ndarray)) else final_tn
        tp = final_tp[i] if isinstance(final_tp, (list, np.ndarray)) else final_tp
        
        frr = fn / (fn + tp + 1e-7)  # False Rejection Rate
        far = fp / (fp + tn + 1e-7)  # False Acceptance Rate
        
        frr_history.append(frr)
        far_history.append(far)
    
    # Find EER
    diff = np.abs(np.array(far_history) - np.array(frr_history))
    eer_idx = np.argmin(diff)
    eer = (far_history[eer_idx] + frr_history[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold, far_history, frr_history


def create_plots_metrics(train_metrics_history, eval_metrics_history):
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
                    'False Negatives', 'False Negatives Value') 



def save_model(model_save_dir_path, dl_model):
    Path(model_save_dir_path).mkdir(parents=True, exist_ok=True)

    dl_model.save(f"{model_save_dir_path}/model_with_weights.h5", save_format='h5')
    print(f"Prediction model saved: {model_save_dir_path}/model_with_weights.h5")



    

def save_metadata(save_path_dir, metadata):
    metadata_path=f"{save_path_dir}/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved: {metadata_path}")

def calculate_metrics(predictions, truePositives, falseNegatives, trueNegatives, falsePositives):
    acc = (truePositives + trueNegatives) / predictions
    precision = truePositives / (truePositives + falsePositives)
    recall = truePositives / (truePositives + falseNegatives)
    f1_score = 2 * (precision * recall)/(precision + recall)

    return acc, precision, recall, f1_score