# First, install Optuna if not already installed
!pip install optuna

import optuna
import subprocess
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import display

# Define your dataset variable
dataset = "FR_Reduced_2K"  # Replace with your actual dataset name

# Create a directory to save results
!mkdir -p optuna_results

# Initialize results tracking
results = []

def objective(trial):
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [1024, 1906, 2048, 4096])
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
    dimension = trial.suggest_categorical("dimension", [50, 100, 200])
    # Use static margin value of 5
    margin = 5
    negative_samples_ratio = trial.suggest_int("negative_samples_ratio", 5, 15)
    regularizer_weight = trial.suggest_float("regularizer_weight", 10.0, 100.0)
    
    # Construct the command
    cmd = f"python Kelpie_hybrid-DB100K/Kelpie_hybrid-DB100K/src/embeddings/transe/train.py "\
          f"--dataset {dataset} "\
          f"--max_epochs 100 "\
          f"--batch_size {batch_size} "\
          f"--learning_rate {learning_rate:.6f} "\
          f"--dimension {dimension} "\
          f"--margin {margin} "\
          f"--negative_samples_ratio {negative_samples_ratio} "\
          f"--regularizer_weight {regularizer_weight:.1f} "\
          f"--valid 5 "\
          f"--verbose False"
    
    print(f"\nTrial #{trial.number}:")
    print(f"Parameters: batch_size={batch_size}, learning_rate={learning_rate:.6f}, dimension={dimension}, "
          f"margin={margin}, negative_samples_ratio={negative_samples_ratio}, "
          f"regularizer_weight={regularizer_weight:.1f}")
    
    # Execute the command and capture output
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = process.stdout
    
    # Check if there was an error
    if process.returncode != 0:
        print("Error executing command!")
        print(process.stderr)
        return 0.0
    
    # Extract performance metrics using regex
    try:
        hits1 = float(re.search(r"Test Hits@1: ([\d\.]+)", output).group(1))
        hits10 = float(re.search(r"Test Hits@10: ([\d\.]+)", output).group(1))
        mrr = float(re.search(r"Test Mean Reciprocal Rank: ([\d\.]+)", output).group(1))
        mean_rank = float(re.search(r"Test Mean Rank: ([\d\.]+)", output).group(1))
    except (AttributeError, ValueError) as e:
        print(f"Failed to extract metrics from output: {e}")
        print(f"Output: {output}")
        return 0.0
    
    # Store results for this trial
    result = {
        "trial": trial.number,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dimension": dimension,
        "margin": margin,
        "negative_samples_ratio": negative_samples_ratio,
        "regularizer_weight": regularizer_weight,
        "hits@1": hits1,
        "hits@10": hits10,
        "mrr": mrr,
        "mean_rank": mean_rank
    }
    results.append(result)
    
    # Save current results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("optuna_results/results.csv", index=False)
    
    # Display current trial results
    print(f"Results: Hits@1={hits1:.4f}, Hits@10={hits10:.4f}, MRR={mrr:.4f}, Mean Rank={mean_rank:.2f}")
    
    # Display current best results based on Hits@1
    best_trial = results_df.loc[results_df['hits@1'].idxmax()]
    print(f"\nCurrent best (Trial #{int(best_trial['trial'])}): Hits@1={best_trial['hits@1']:.4f}")
    
    return hits1  # Now optimizing for Hits@1 instead of MRR

# Create and run the study
study = optuna.create_study(direction="maximize", study_name="TransE-Optimization-Hits1")
study.optimize(objective, n_trials=15)  # Adjust number of trials as needed

# Display final results
print("\n==== OPTIMIZATION COMPLETE ====")
print(f"Best parameters: {study.best_params}")
print(f"Best performance (Hits@1): {study.best_value:.4f}")

# Final results dataframe
final_df = pd.DataFrame(results)
final_df.to_csv("optuna_results/final_results.csv", index=False)

# Display top 5 configurations sorted by Hits@1
top_configs = final_df.sort_values("hits@1", ascending=False).head(5)
print("\nTop 5 configurations (by Hits@1):")
display(top_configs)

# Create visualization plots
try:
    # Plot optimization history
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig("optuna_results/optimization_history.png")
    plt.figure()
    
    # Plot parameter importances
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig("optuna_results/param_importances.png")
    plt.figure()
    
    # Plot parallel coordinate plot
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig("optuna_results/parallel_coordinate.png")
    
    print("Visualization plots saved to optuna_results directory")
except Exception as e:
    print(f"Error creating visualization plots: {e}")

# Generate command for best parameters
best_params = study.best_params
best_cmd = f"!python Kelpie_hybrid-DB100K/Kelpie_hybrid-DB100K/src/embeddings/transe/train.py "\
          f"--dataset {dataset} "\
          f"--max_epochs 100 "\
          f"--batch_size {best_params['batch_size']} "\
          f"--learning_rate {best_params['learning_rate']:.6f} "\
          f"--dimension {best_params['dimension']} "\
          f"--margin 5 "\
          f"--negative_samples_ratio {best_params['negative_samples_ratio']} "\
          f"--regularizer_weight {best_params['regularizer_weight']:.1f} "\
          f"--valid 5 "\
          f"--verbose False"

print("\nCommand with best parameters:")
print(best_cmd)
