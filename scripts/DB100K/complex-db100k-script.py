#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

def run_pipeline(config, builder):
    # Extract parameters from config
    dataset = config.get("dataset", "DB100K")
    embedding_model = config.get("embedding_model", "ComplEx")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    predictions = config.get("predictions", "transe_db100k_random.csv")
    rules_file = config.get("rules_file", "DB100K.csv")
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    editorial_rules = config.get("editorial_rules", None)
    dimension = config.get("dimension", 80)
    verify_epochs = config.get("verify_epochs", 10)
    max_epochs = config.get("max_epochs", 50)
    pca = config.get("pca", 0.2)

    # Learning parameters
    batch_size = config.get("batch_size", 12)
    learning_rate = config.get("learning_rate", 0.1)
    reg = config.get("reg", "5e-3")
    optimizer = config.get("optimizer", "Adagrad")
    mode = config.get("mode", "necessary")
    
    # Create exp_id based on configuration
    exp_id = config.get("exp_id", f"_complex_100_pca_{pca}")
    message = config.get("message", f"[experiments] complex 100 pca {pca}")

    # Print configuration
    print(f"Running with builder: {builder}, PCA: {pca}")
    print("dataset: ", dataset)
    print("stored_model: ", stored_model)
    print("predictions: ", predictions)
    print("rules_files: ", rules_file)
    print("explanations: ", explanations)

    # Create logging directory
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"data/7_logging/{current_date}"
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamped log file names
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    explain_log_file = f"{log_dir}/{current_timestamp}_PRINT_explain_{builder}_{predictions}.txt"
    verify_log_file = f"{log_dir}/{current_timestamp}_PRINT_verify_{builder}_{predictions}.txt"

    # Run explanation process
    explain_cmd = [
        "python", "../../src/embeddings/complex/explain.py",
        "--pca", str(pca),
        "--builder", builder,
        "--dataset", dataset,
        "--max_epochs", str(max_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--dimension", str(dimension),
        "--optimizer", optimizer,
        "--reg", reg,
        "--model_file", stored_model,
        "--predictions_to_explain_file", predictions,
        "--mode", mode,
        "--verbose", "False",
        "--rules_file", rules_file
    ]

    # Add editorial_rules if specified
    if editorial_rules:
        explain_cmd.extend(["--second_rules_file", editorial_rules])

    print(f"Running explanation process with builder: {builder}...")
    with open(explain_log_file, 'w') as log_file:
        subprocess.run(explain_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Run verification process
    verify_cmd = [
        "python", "../../src/embeddings/complex/verify_explanations.py",
        "--dataset", dataset,
        "--max_epochs", str(verify_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--dimension", str(dimension),
        "--optimizer", optimizer,
        "--reg", reg,
        "--model_file", stored_model,
        "--mode", mode,
        "--verbose", "True"
    ]

    print(f"Running verification process for builder: {builder}...")
    with open(verify_log_file, 'w') as log_file:
        subprocess.run(verify_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"Process completed successfully for builder: {builder}")
    print(f"Logs saved to: {explain_log_file} and {verify_log_file}")
    print("-----------------------------------")

def main():
    # Read configuration from input.json
    try:
        with open('input_db100k_complex.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: input_complex_db100k.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: input_complex_db100k.json contains invalid JSON.")
        return

    # Get list of builders to run
    builders = config.get("builders", ["rules_kelpie"])
    # If builders is a string, convert to list
    if isinstance(builders, str):
        builders = [builders]
    
    # Run pipeline for each builder
    for builder in builders:
        run_pipeline(config, builder)

if __name__ == "__main__":
    main()
