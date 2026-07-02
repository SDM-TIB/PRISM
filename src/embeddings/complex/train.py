import argparse
import numpy
import os
import sys
import torch

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import MODEL_PATH, ALL_DATASET_NAMES
from src.link_prediction.evaluation.evaluation import Evaluator
from src.link_prediction.models.complex import ComplEx
from src.link_prediction.models.model import OPTIMIZER_NAME, LEARNING_RATE, REGULARIZER_NAME, REGULARIZER_WEIGHT, BATCH_SIZE, DECAY_1, DECAY_2, DIMENSION, INIT_SCALE, EPOCHS
from src.link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from src.model.dataset import Dataset
from src.model.utils import generate_output_file_prefix

EMBEDDING = "ComplEx"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Relational learning contraption"
    )

    parser.add_argument('--dataset',
                        choices=ALL_DATASET_NAMES,
                        default="FR_Reduced_2K",
                        help="Dataset in {}".format(ALL_DATASET_NAMES)
    )

    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument('--optimizer',
                        choices=optimizers,
                        default='Adagrad',
                        help="Optimizer in {}".format(optimizers)
    )

    parser.add_argument('--max_epochs',
                        default=50,
                        type=int,
                        help="Number of epochs."
    )

    parser.add_argument('--valid',
                        default=10,
                        type=float,
                        help="Number of epochs before valid."
    )

    parser.add_argument('--dimension',
                        default=1000,
                        type=int,
                        help="Embedding dimension"
    )

    parser.add_argument('--batch_size',
                        default=1000,
                        type=int,
                        help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
    )

    parser.add_argument('--reg',
                        default=5e-3,   #Y 5e-3     #WN 5e-2    #FB 2.5e-3
                        type=float,
                        help="Regularization weight"
    )

    parser.add_argument('--init_scale',
                        default=1e-3,
                        type=float,
                        help="Initial scale"
    )

    parser.add_argument('--learning_rate',
                        default=0.1,    #Y 0.01
                        type=float,
                        help="Learning rate"
    )

    parser.add_argument('--decay1',
                        default=0.9,
                        type=float,
                        help="Decay rate for the first moment estimate in Adam"
    )

    parser.add_argument('--decay2',
                        default=0.999,
                        type=float,
                        help="Decay rate for second moment estimate in Adam"
    )

    parser.add_argument('--load',
                        help="path to the model to load",
                        required=False)

    parser.add_argument("--verbose",
                        type=bool,
                        default=True,
                        help="extend of print output")

    args = parser.parse_args()

    #deterministic!
    seed = 42
    print(f"using seed {seed}")
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
    dataset = Dataset(name=dataset_name, separator="\t", load=True)

    if args.load is not None:
        model_path = args.load
    else:
        model_path = os.path.join(MODEL_PATH, "_".join([EMBEDDING, dataset_name]) + ".pt")
        if not os.path.isdir(MODEL_PATH):
            os.makedirs(MODEL_PATH)

    hyperparameters = {DIMENSION:args.dimension,
                    INIT_SCALE:args.init_scale,
                    OPTIMIZER_NAME:args.optimizer,
                    BATCH_SIZE:args.batch_size,
                    EPOCHS:args.max_epochs,
                    LEARNING_RATE:args.learning_rate,
                    DECAY_1:args.decay1,
                    DECAY_2:args.decay2,
                    REGULARIZER_NAME:'N3',
                    REGULARIZER_WEIGHT:args.reg}

    output_prefix, timestamp = generate_output_file_prefix(embedding=EMBEDDING)

    filename_train_log = output_prefix + "_1_train_py.csv"
    print(filename_train_log)
    with open(filename_train_log, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"train.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, model: {model_path}\n")
        
    print("Initializing model...")
    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx

    print(f"Training model... {EMBEDDING}")
    optimizer = MultiClassNLLOptimizer(model=model,
                                    hyperparameters=hyperparameters, verbose=False)

    optimizer.train(train_samples=dataset.train_samples,
                    save_path=model_path,
                    evaluate_every=args.valid,
                    valid_samples=dataset.valid_samples)

    print("Evaluating model...")
    mrr, h1, h10, mr = Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=False)      #produces filtered_ranks.csv
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
