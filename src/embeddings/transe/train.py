import argparse
import numpy
import os
import random
import sys
import torch

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import MODEL_PATH, ALL_DATASET_NAMES
from src.link_prediction.evaluation.evaluation import Evaluator
from src.link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT
from src.link_prediction.models.transe import TransE
from src.link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from src.model.dataset import Dataset
from src.model.utils import generate_output_file_prefix

EMBEDDING = "TransE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        default="FR_Reduced_2K",
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

    parser.add_argument("--valid",
                        type=int,
                        default=20,
                        help="Validate after a cycle of x epochs")

    parser.add_argument("--verbose",
                        type=bool,
                        default=True,
                        help="extend of print output")

    args = parser.parse_args()
    
    #deterministic!
    seed = 42
    print(f"using seed {seed}")
    random.seed(seed)   #TODO: why only for this model?
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    kernel = ""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        kernel = "cuda"
    else:
        kernel = "cpu"
    print("kernel: ", kernel)


    print("Loading %s dataset..." % args.dataset)
    dataset_name = args.dataset
    dataset = Dataset(dataset_name)

    model_path=os.path.join(MODEL_PATH, EMBEDDING + dataset_name + ".pt")
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    hyperparameters = {DIMENSION: args.dimension,
                       MARGIN: args.margin,
                       NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.regularizer_weight,
                       BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate,
                       EPOCHS: args.max_epochs}

    output_prefix, timestamp = generate_output_file_prefix(embedding=EMBEDDING)

    filename_train_log = output_prefix + "_1_train_py.csv"
    print(filename_train_log)
    with open(filename_train_log, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"train.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, model: {model_path}\n")

    print("Initializing model...")
    transe = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: TransE

    print(f"Training model... {EMBEDDING}")
    optimizer = PairwiseRankingOptimizer(model=transe, hyperparameters=hyperparameters, verbose=False)

    optimizer.train(train_samples=dataset.train_samples, evaluate_every=args.valid,
                    save_path=model_path,
                    valid_samples=dataset.valid_samples)

    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=transe, output_prefix=output_prefix).evaluate(samples=dataset.test_samples, write_output=False)      #produces filtered_ranks.csv
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

    stats = f"{h1}; {h10}; {mrr}; {mr}; {args.dimension}; {args.margin}; {args.negative_samples_ratio}; {args.regularizer_weight}; {args.batch_size}; {args.learning_rate}; {args.max_epochs}\n"

    filename_stats = output_prefix + "_2_training_stats.csv"
    print(filename_stats)
    with open(filename_stats, "w", encoding="utf8" ) as training_stats:
        training_stats.write(stats)
    
    print("model path:\n", model_path)
