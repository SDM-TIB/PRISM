import argparse
import numpy
import os
import sys
import torch

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import MODEL_PATH, LOGGING_PATH, ALL_DATASET_NAMES
from src.model.utils import generate_output_file_prefix
from src.model.dataset import Dataset
from src.link_prediction.evaluation.evaluation import Evaluator
from src.link_prediction.models.transe import TransE
from src.link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT

EMBEDDING = "TransE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        default="FR_Reduced_2K_copy",
                        choices=ALL_DATASET_NAMES,
                        help=f"The dataset to use: {ALL_DATASET_NAMES}")

    parser.add_argument("--max_epochs",
                        type=int,
                        default=100,
                        help="Number of epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=2048,
                        help="Batch size.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate.")

    parser.add_argument("--dimension",
                        type=int,
                        default=200,
                        help="Embedding dimensionality.")

    parser.add_argument("--margin",
                        type=int,
                        default=5,
                        help="Margin for pairwise ranking loss.")

    parser.add_argument("--negative_samples_ratio",
                        type=int,
                        default=10,
                        help="Number of negative samples for each positive sample.")

    parser.add_argument("--regularizer_weight",
                        type=float,
                        default=50.0,
                        help="Weight for L2 regularization.")

    parser.add_argument("--model_file",
                        type=str,
                        default="TransE_FR_Reduced_2K_copy.pt",
                        help="The path where the model can be found")

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_path = os.path.join(MODEL_PATH, args.model_file)
    dataset_name = args.dataset
    dataset = Dataset(dataset_name)

    hyperparameters = {DIMENSION: args.dimension,
                       MARGIN: args.margin,
                       NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.regularizer_weight,
                       BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate,
                       EPOCHS: args.max_epochs}

    print(f"Initializing model... {EMBEDDING}")
    transe = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: TransE
    kernel = ""
    if torch.cuda.is_available():
        kernel = "cuda"
        transe.to(kernel)
        transe.load_state_dict(torch.load(model_path))
    else:
        kernel = "cpu"
        transe.load_state_dict(torch.load(model_path, map_location=torch.device(kernel)))
    transe.eval()

    output_prefix, timestamp = generate_output_file_prefix(embedding=EMBEDDING)

    filename = os.path.join(LOGGING_PATH, output_prefix + "_3_test_py.txt")
    print(filename)
    with open(filename, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"train.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}\n")


    print("\nEvaluating model...")
    mrr, h1, h10, mr = Evaluator(model=transe, output_prefix=output_prefix).evaluate(samples=dataset.test_samples, write_output=True)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

    with open(filename, "a", encoding="utf8" ) as execution_log:
        execution_log.write(f"Evaluating model\nTest Hits@1: {h1}, Test Hits@10:: {h10}, Test Mean Reciprocal Rank: {mrr}, Test Mean Rank: {mr}, \n")