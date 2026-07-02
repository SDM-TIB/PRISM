import argparse
import os
import sys
import torch

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import LOGGING_PATH, MODEL_PATH, ALL_DATASET_NAMES
from model.utils import generate_output_file_prefix
from src.model.dataset import Dataset
from src.link_prediction.evaluation.evaluation import Evaluator
from src.link_prediction.models.complex import ComplEx
from src.link_prediction.models.model import DIMENSION, INIT_SCALE

EMBEDDING = "ComplEx"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Kelpie")

    parser.add_argument('--dataset',
                        choices=ALL_DATASET_NAMES,
                        default="FR_Reduced_2K",
                        help="Dataset in {}".format(ALL_DATASET_NAMES))

    parser.add_argument('--dimension',
                        default=1000,
                        type=int,
                        help="Embedding dimension")

    parser.add_argument('--init_scale',
                        default=0.001,
                        type=float,
                        help="Initial scale")

    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help="Learning rate")

    parser.add_argument('--model_file',# TODO: rename everywhere to model__file?
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
                        help="path to the model to load",
                        required=True)

    args = parser.parse_args()
    model_path = os.path.join(MODEL_PATH, args.model_file)

    # Deterministic (not originally in complex/train)
    torch.backends.cudnn.deterministic = True
    seed = 42
    torch.manual_seed(seed)

    print("Loading %s dataset..." % args.dataset)
    dataset = Dataset(name=args.dataset, separator="\t", load=True)

    hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init_scale}
    print("Initializing model...")
    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx

    kernel = ""
    if torch.cuda.is_available():
        kernel = "cuda"
        torch.cuda.manual_seed_all(seed)
        model.to(kernel)
        model.load_state_dict(torch.load(model_path))
    else:
        kernel = "cpu"
        model.load_state_dict(torch.load(model_path, map_location=torch.device(kernel)))
    print("kernel: ", kernel)

    model.eval()

    output_prefix, timestamp = generate_output_file_prefix(embedding=EMBEDDING)

    filename = os.path.join(LOGGING_PATH, output_prefix + "_3_test_py.txt")
    print(filename)
    with open(filename, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"train.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}\n")

    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=model, output_prefix=output_prefix).evaluate(samples=dataset.test_samples, write_output=True)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Hits@10: %f" % h10)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)
    print("\tTest Mean Rank: %f" % mr)

