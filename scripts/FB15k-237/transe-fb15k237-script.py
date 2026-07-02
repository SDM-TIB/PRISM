#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

def main():
    # Read configuration from input.json
    try:
        with open('input_fb15k237_transe.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: input.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: input.json contains invalid JSON.")
        return

    # Extract parameters from config
    dataset = config.get("dataset", "FB15k-237")
    embedding_model = config.get("embedding_model", "TransE")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    filtered_ranks = config.get("filtered_ranks", f"{embedding_model}_{dataset}_filtered_ranks.csv")
    predictions = config.get("predictions", f"{embedding_model.lower()}_fb15k237_few.csv")
    rules_file = config.get("rules_file", "FreeBase.csv")
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    editorial_rules = config.get("editorial_rules", None)
    dimension = config.get("dimension", 50)
    verify_epochs = config.get("verify_epochs", 10)
    max_epochs = config.get("max_epochs", 100)
    pca_threshold = config.get("pca_threshold", 0.7)

    # Learning parameters
    batch_size = config.get("batch_size", 2048)
    learning_rate = config.get("learning_rate", 0.0004)
    negative_samples_ratio = config.get("negative_samples_ratio", 15)
    regularizer_weight = config.get("regularizer_weight", 1.0)
    valid = config.get("valid", 10)
    mode = config.get("mode", "sufficient")

    exp_id = config.get("exp_id", "_transE100")  # must start with _
    message = config.get("message", "[experiments] transe 100")

    # Print configuration
    print("dataset: ", dataset)
    print("stored_model: ", stored_model)
    print("filtered_ranks: ", filtered_ranks)
    print("predictions: ", predictions)
    print("rules_files: ", rules_file, ", ", editorial_rules)
    print("explanations: ", explanations)

    # Create logging directory
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"data/7_logging/{current_date}"
    os.makedirs(log_dir, exist_ok=True)

    # Set builder
    builder = config.get("builder","kelpie")

    # Generate timestamped log file names
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    explain_log_file = f"{log_dir}/{current_timestamp}_PRINT_explain_{builder}_{predictions}.txt"
    verify_log_file = f"{log_dir}/{current_timestamp}_PRINT_verify_{builder}_{predictions}.txt"

    # Run explanation process
    explain_cmd = [
        "python", "../../src/embeddings/transe/explain.py",
        "--builder", builder,
        "--dataset", dataset,
        "--max_epochs", str(max_epochs),
        "--batch_size", str(batch_size),
	"--rules_file", str(rules_file),
        "--learning_rate", str(learning_rate),
        "--dimension", str(dimension),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--regularizer_weight", str(regularizer_weight),
        "--model_file", stored_model,
        "--predictions_to_explain_file", predictions,
        "--mode", mode,
        "--verbose", "False",
        "--pca", str(pca_threshold)
    ]

    # Add editorial_rules if specified
    if editorial_rules:
        explain_cmd.extend(["--second_rules_file", editorial_rules])

    #print("Running explanation process...")
    with open(explain_log_file, 'w') as log_file:
        subprocess.run(explain_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Run verification process
    verify_cmd = [
        "python", "../../src/embeddings/transe/verify_explanations.py",
        "--dataset", dataset,
        "--max_epochs", str(verify_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--dimension", str(dimension),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--regularizer_weigh", str(regularizer_weight),
        "--model_file", stored_model,
        "--mode", mode,
        "--verbose", "False"
    ]

    print("Running verification process...")
    with open(verify_log_file, 'w') as log_file:
        subprocess.run(verify_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"Process completed successfully. Message: {message}")
    print(f"Logs saved to: {explain_log_file} and {verify_log_file}")

if __name__ == "__main__":
    main()