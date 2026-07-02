#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

def run_pipeline(config, builder):
    # Extract parameters from config
    dataset = config.get("dataset", "FR_Reduced_2K")
    embedding_model = config.get("embedding_model", "TransE")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    predictions = config.get("predictions", "FR_10.csv")
    rules_file = config.get("rules_file", f"{dataset.lower()}_rules_optimai.csv")
    editorial_rules = config.get("editorial_rules", "FR_editorial_rules.csv")
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    dimension = config.get("dimension", 50)
    batch_size = config.get("batch_size", 1906)
    
    # FR TransE specific settings
    negative_samples_ratio = config.get("negative_samples_ratio", 5)
    regularizer_weight = config.get("regularizer_weight", 50.0)  # Using 50.0 as in the notebook code
    e_regularizer_weight = config.get("e_regularizer_weight", 50.0)
    v_regularizer_weight = config.get("v_regularizer_weight", 2.0)
    margin = config.get("margin", 2)
    
    # Learning parameters
    e_learning_rate = config.get("e_learning_rate", 0.003)  # Using 0.003 as in the notebook code
    v_learning_rate = config.get("v_learning_rate", 0.00003)
    e_epochs = config.get("e_epochs", 100)
    v_epochs = config.get("v_epochs", 10)
    
    thr = config.get("thr", 0.7)
    coverage = config.get("coverage", 3)
    mode = config.get("mode", "sufficient")
    
    exp_id = config.get("exp_id", f"_{embedding_model}_thr_{thr}_{mode[0]}")
    message = config.get("message", f"[experiments] {embedding_model} {predictions} thr:{thr} {mode[0]}")

    # Print configuration
    print(f"Running with builder: {builder}")
    print("dataset: ", dataset)
    print("stored_model: ", stored_model)
    print("predictions: ", predictions)
    print("rules_files: ", rules_file, ", ", editorial_rules)
    print("explanations: ", explanations)

    # Create logging directory
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"data/7_logging/{current_date}"
    os.makedirs(log_dir, exist_ok=True)

    # Builder setup
    current_builder = builder
    if builder == "kelpie":
        current_builder = f"kelpie{exp_id}"

    # Generate timestamped log file names
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    explain_log_file = f"{log_dir}/{current_timestamp}_PRINT_explain_{current_builder}_{predictions}.txt"
    verify_log_file = f"{log_dir}/{current_timestamp}_PRINT_verify_{current_builder}_{predictions}.txt"

    # Run explanation process
    explain_cmd = [
        "python", "../../src/embeddings/transe/explain.py",
        "--dataset", dataset,
        "--max_epochs", str(e_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(e_learning_rate),
        "--dimension", str(dimension),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--regularizer_weight", str(e_regularizer_weight),
        "--margin", str(margin),
        "--model_file", stored_model,
        "--predictions_to_explain_file", predictions,
        "--mode", mode,
        "--verbose", "False",
        "--builder", current_builder,
        "--rules_file", rules_file,
        "--second_rules_file", editorial_rules,
        "--pca", str(thr),
        "--coverage", str(coverage)
    ]

    print(f"Running explanation process with builder: {current_builder}...")
    with open(explain_log_file, 'w') as log_file:
        subprocess.run(explain_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Run verification process
    verify_cmd = [
        "python", "../../src/embeddings/transe/verify_explanations.py",
        "--dataset", dataset,
        "--model_file", stored_model,
        "--dimension", str(dimension),
        "--batch_size", str(batch_size),
        "--max_epochs", str(v_epochs),
        "--learning_rate", str(v_learning_rate),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--margin", str(margin),
        "--regularizer_weight", str(v_regularizer_weight),
        "--mode", mode,
        "--verbose", "False"
    ]

    print(f"Running verification process for builder: {current_builder}...")
    with open(verify_log_file, 'w') as log_file:
        subprocess.run(verify_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"Process completed successfully for builder: {current_builder}. Message: {message}")
    print(f"Logs saved to: {explain_log_file} and {verify_log_file}")
    print("-----------------------------------")

def main():
    # Read configuration from input.json
    try:
        with open('input_transe_fr.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: input_transe_fr.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: input_transe_fr.json contains invalid JSON.")
        return

    # Get list of builders to run
    builders = config.get("builders", ["kelpie"])
    # If builders is a string, convert to list
    if isinstance(builders, str):
        builders = [builders]
    
    # Run pipeline for each builder
    for builder in builders:
        run_pipeline(config, builder)

if __name__ == "__main__":
    main()
