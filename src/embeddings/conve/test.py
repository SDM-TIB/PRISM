import argparse
import numpy
import os
import sys
import torch

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import MODEL_PATH, LOGGING_PATH, ALL_DATASET_NAMES
from src.link_prediction.evaluation.evaluation import Evaluator
from src.link_prediction.models.conve import ConvE
from src.link_prediction.models.model import INPUT_DROPOUT, BATCH_SIZE, LEARNING_RATE, DECAY, LABEL_SMOOTHING, \
    EPOCHS, DIMENSION, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE
from src.model.dataset import Dataset
from src.model.utils import generate_output_file_prefix

EMBEDDING = "ConvE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        help=f"The dataset to use: {ALL_DATASET_NAMES}")

    parser.add_argument("--max_epochs",
                        type=int,
                        default=1000,
                        help="Number of epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0005,
                        help="Learning rate.")

    parser.add_argument("--decay_rate",
                        type=float,
                        default=1.0,
                        help="Decay rate.")

    parser.add_argument("--dimension",
                        type=int,
                        default=200,
                        help="Embedding dimensionality.")

    parser.add_argument("--valid",
                        type=int,
                        default=-1,
                        help="Validate after a cycle of x epochs")

    parser.add_argument("--input_dropout",
                        type=float,
                        default=0.3,
                        nargs="?",
                        help="Input layer dropout.")

    parser.add_argument("--hidden_dropout",
                        type=float,
                        default=0.4,
                        help="Dropout after the hidden layer.")

    parser.add_argument("--feature_map_dropout",
                        type=float,
                        default=0.5,
                        help="Dropout after the convolutional layer.")

    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        help="Amount of label smoothing.")

    parser.add_argument('--hidden_size',
                        type=int,
                        default=9728,
                        help='The side of the hidden layer. '
                             'The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

    parser.add_argument('--model_file',
                        type=str,
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
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
                       INPUT_DROPOUT: args.input_dropout,
                       FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout,
                       HIDDEN_LAYER_SIZE: args.hidden_size,
                       BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate,
                       DECAY: args.decay_rate,
                       LABEL_SMOOTHING: args.label_smoothing,
                       EPOCHS: args.max_epochs}


    print(f"Initializing model... {EMBEDDING}")
    model = ConvE(dataset=dataset,
                  hyperparameters=hyperparameters,
                  init_random=True) # type: ConvE
    kernel = ""
    if torch.cuda.is_available():
        kernel = "cuda"
        model.to(kernel)
        model.load_state_dict(torch.load(model_path))
    else:
        kernel = "cpu"
        model.load_state_dict(torch.load(model_path, map_location=torch.device(kernel)))
    model.eval()

    output_prefix, timestamp = generate_output_file_prefix(embedding=EMBEDDING)

    filename = os.path.join(LOGGING_PATH, output_prefix + "_3_test_py.txt")
    print(filename)
    with open(filename, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"train.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}\n")
    
    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=True)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

    with open(filename, "a", encoding="utf8" ) as execution_log:
        execution_log.write(f"Evaluating model\nTest Hits@1: {h1}, Test Hits@10:: {h10}, Test Mean Reciprocal Rank: {mrr}, Test Mean Rank: {mr}, \n")
