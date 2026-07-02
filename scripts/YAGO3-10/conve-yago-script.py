#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

def main():
    # Read configuration from input.json
    try:
        with open('input_YAGO_ConvE.json', 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Error: input.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: input.json contains invalid JSON.")
        return

    # Extract parameters from config
    dataset = config.get("dataset", "YAGO3-10")
    embedding_model = config.get("embedding_model", "ConvE")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    predictions = config.get("predictions", "transe_yago_daniel.csv")
    rules_file = config.get("rules_file", f"{dataset.lower()}_optimai.csv")
    editorial_rules = config.get("editorial_rules", None)
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    dimension = config.get("dimension", 100)
    batch_size = config.get("batch_size", 2048)

    # ConvE specific settings
    input_dropout = config.get("input_dropout", 0.2)
    hidden_dropout = config.get("hidden_dropout", 0.3)
    feature_map_dropout = config.get("feature_map_dropout", 0.2)
    decay_rate = config.get("decay_rate", 0.995)

    # Learning parameters
    e_learning_rate = config.get("e_learning_rate", 0.003)
    v_learning_rate = config.get("v_learning_rate", 0.003)
    e_epochs = config.get("e_epochs", 100)
    v_epochs = config.get("v_epochs", 10)

    thr = config.get("thr", 0.7)
    coverage = config.get("coverage", 3)

    mode = config.get("mode", "sufficient")  # Alternative: "necessary"

    exp_id = config.get("exp_id", f"_{embedding_model}_thr_{thr}_{mode[0]}")  # must start with _
    message = config.get("message", f"[experiments] {embedding_model} {predictions} thr:{thr} {mode[0]}")

    # Print configuration
    print("dataset: ", dataset)
    print("stored_model: ", stored_model)
    print("predictions: ", predictions)
    print("rules_files: ", rules_file, ", ", editorial_rules)
    print("explanations: ", explanations)

    # Create logging directory
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"data/7_logging/{current_date}/{embedding_model}"
    os.makedirs(log_dir, exist_ok=True)

    # Set builder
    builder = config.get("builder", "Kelpie")

    # Run explanation process
    explain_cmd = [
        "python", "../../src/embeddings/conve/explain.py",
        "--dataset", dataset,
        "--max_epochs", str(e_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(e_learning_rate),
        "--dimension", str(dimension),
        "--input_dropout", str(input_dropout),
        "--hidden_dropout", str(hidden_dropout),
        "--feature_map_dropout", str(feature_map_dropout),
        "--decay_rate", str(decay_rate),
        "--model_file", stored_model,
        "--predictions_to_explain_file", predictions,
        "--mode", mode,
        "--verbose", "False",
        "--builder", builder,
        "--rules_file", rules_file,
        "--pca", str(thr),
        "--coverage", str(coverage)
    ]

    # Add editorial_rules if specified
    if editorial_rules:
        explain_cmd.extend(["--second_rules_file", editorial_rules])

    print("Running explanation process...")
    subprocess.run(explain_cmd)

    # Run verification process
    verify_cmd = [
        "python", "../../src/embeddings/conve/verify_explanations.py",
        "--dataset", dataset,
        "--model_file", stored_model,
        "--dimension", str(dimension),
        "--batch_size", str(batch_size),
        "--max_epochs", str(v_epochs),
        "--learning_rate", str(v_learning_rate),
        "--input_dropout", str(input_dropout),
        "--hidden_dropout", str(hidden_dropout),
        "--feature_map_dropout", str(feature_map_dropout),
        "--decay_rate", str(decay_rate),
        "--mode", mode,
        "--verbose", "False"
    ]

    print("Running verification process...")
    subprocess.run(verify_cmd)

    print(f"Process completed successfully. Message: {message}")

if __name__ == "__main__":
    main()
