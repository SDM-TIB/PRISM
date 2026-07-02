#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

def run_pipeline(config, builder):
    # Extract parameters from config
    dataset = config.get("dataset", "YAGO3-10")
    embedding_model = config.get("embedding_model", "ComplEx")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    predictions = config.get("predictions", "transe_yago_daniel.csv")
    rules_file = config.get("rules_file", f"{dataset.lower()}_optimai.csv")
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    editorial_rules = config.get("editorial_rules", None)
    dimension = config.get("dimension", 1000)
    verify_epochs = config.get("verify_epochs", 10)
    max_epochs = config.get("max_epochs", 100)
    pca_threshold = config.get("pca_threshold", 0.7)

    # Learning parameters
    batch_size = config.get("batch_size", 1000)
    learning_rate = config.get("learning_rate", 0.1)
    reg = config.get("reg", "5e-2")
    optimizer = config.get("optimizer", "Adagrad")
    mode = config.get("mode", "necessary")
    
    exp_id = config.get("exp_id", "_complEx100")  # must start with _
    message = config.get("message", f"[experiments] complex {builder}")

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
        "python", "../../src/embeddings/complex/explain.py",
        "--builder", current_builder,
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
        "--verbose", "False"
    ]

    # Add rules_file if specified
    if rules_file:
        explain_cmd.extend(["--rules_file", rules_file])

    # Add pca threshold if this is a pca or frequency builder
    if builder == "pca" or "pca" in builder or builder == "frequency" or "frequency" in builder:
        explain_cmd.extend(["--pca", str(pca_threshold)])

    # Add editorial_rules if specified
    if editorial_rules:
        explain_cmd.extend(["--second_rules_file", editorial_rules])

    print(f"Running explanation process with builder: {current_builder}...")
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
        with open('input_fb15k237_complex.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: input_fb15k237_complex.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: input_fb15k237_complex.json contains invalid JSON.")
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
