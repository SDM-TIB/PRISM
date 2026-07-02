import json
import os
import subprocess
from datetime import datetime

def run_commands(config):
    # Extract parameters from config
    dataset = config.get("dataset", "YAGO3-10")
    embedding_model = config.get("embedding_model", "TransE")
    stored_model = config.get("stored_model", f"{embedding_model}_{dataset}.pt")
    predictions = config.get("predictions", f"transe_yago_daniel.csv")
    rules_file = config.get("rules_file", f"{dataset.lower()}_optimai.csv")
    editorial_rules = config.get("editorial_rules", None)
    explanations = config.get("explanations", f"{stored_model[:-3]}_explanation.csv")
    dimension = config.get("dimension", 100)
    batch_size = config.get("batch_size", 2048)
    negative_samples_ratio = config.get("negative_samples_ratio", 10)
    regularizer_weight = config.get("regularizer_weight", 50)
    margin = config.get("margin", 5)
    e_learning_rate = config.get("e_learning_rate", 0.01)
    v_learning_rate = config.get("v_learning_rate", 0.003)
    e_epochs = config.get("e_epochs", 100)
    v_epochs = config.get("v_epochs", 10)
    thr = config.get("thr", 0.7)
    coverage = config.get("coverage", 3)
    mode = config.get("mode", "sufficient")
    
    # Create experiment ID and message
    exp_id = f"_{embedding_model}_thr_{thr}_{mode[0]}"
    message = f"[experiments] {embedding_model} {predictions} thr:{thr} {mode[0]}"
    
    # Print configuration
    print("dataset: ", dataset)
    print("stored_model: ", stored_model)
    print("predictions: ", predictions)
    print("rules_files: ", rules_file, ", ", editorial_rules)
    print("explanations: ", explanations)
    
    # Create directories for logging
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"data/7_logging/{today}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set builder
    builder = f"kelpie_rules"
    
    # Command for explain.py
    explain_cmd = [
        "python", "../../src/embeddings/transe/explain.py",
        "--dataset", dataset,
        "--max_epochs", str(e_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(e_learning_rate),
        "--dimension", str(dimension),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--regularizer_weight", str(regularizer_weight),
        "--margin", str(margin),
        "--model_file", stored_model,
        "--predictions_to_explain_file", predictions,
        "--mode", mode,
        "--verbose", "False",
        "--builder", builder,
        "--rules_file", rules_file,
        "--pca", str(thr),
        "--coverage", str(coverage)
    ]
    
    # Add editorial_rules if provided
    if editorial_rules:
        explain_cmd.extend(["--second_rules_file", editorial_rules])
    
    # Command for verify_explanations.py
    verify_cmd = [
        "python", "../../src/embeddings/transe/verify_explanations.py",
        "--dataset", dataset,
        "--model_file", stored_model,
        "--dimension", str(dimension),
        "--batch_size", str(batch_size),
        "--max_epochs", str(v_epochs),
        "--learning_rate", str(v_learning_rate),
        "--negative_samples_ratio", str(negative_samples_ratio),
        "--regularizer_weight", str(regularizer_weight),
        "--margin", str(margin),
        "--mode", mode,
        "--verbose", "False"
    ]
    
    # Execute commands
    print(f"\nExecuting: {' '.join(explain_cmd)}")
    subprocess.run(explain_cmd)
    
    print(f"\nExecuting: {' '.join(verify_cmd)}")
    subprocess.run(verify_cmd)

def main():
    # Parse input.json file
    try:
        with open('input_YAGO_transe.json', 'r') as f:
            config = json.load(f)
        
        # Run the commands with the loaded config
        run_commands(config)
    except FileNotFoundError:
        print("Error: input.json file not found.")
    except json.JSONDecodeError:
        print("Error: input.json is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()