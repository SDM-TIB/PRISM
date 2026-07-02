import argparse
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from src.embeddings.unified_verify_explanations import verify
from config import ALL_DATASET_NAMES

EMBEDDING = "ComplEx"

def main():
    datasets = ALL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

    parser.add_argument('--dataset',
                        choices=datasets,
                        default="FR_Reduced_2K",
                        help="Dataset in {}".format(datasets),
                        )#required=True)

    parser.add_argument('--model_file',
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
                        help="Path to the model to explain the predictions of",
                        )#required=True)

    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument('--optimizer',
                        choices=optimizers,
                        default='Adagrad',
                        help="Optimizer in {} to use in post-training".format(optimizers))

    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help="Batch size to use in post-training")

    parser.add_argument('--max_epochs',
                        default=10,#200,
                        type=int,
                        help="Number of epochs to run in post-training")

    parser.add_argument('--dimension',
                        default=1000,
                        type=int,
                        help="Factorization rank.")

    parser.add_argument('--learning_rate',
                        default=1e-1,
                        type=float,
                        help="Learning rate")

    parser.add_argument('--reg',
                        default=0,
                        type=float,
                        help="Regularization weight")

    parser.add_argument('--init',
                        default=1e-3,
                        type=float,
                        help="Initial scale")

    parser.add_argument('--decay1',
                        default=0.9,
                        type=float,
                        help="Decay rate for the first moment estimate in Adam")

    parser.add_argument('--decay2',
                        default=0.999,
                        type=float,
                        help="Decay rate for second moment estimate in Adam")

    parser.add_argument("--mode",
                        type=str,
                        default="sufficient",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    # parser.add_argument("--explanations_to_verify_file",
    #                     type=str,
    #                     default=f"TransE_FR_Reduced_2K_explanation.csv",
    #                     help="The file with facts which shall be explained")
    
    parser.add_argument("--verbose",
                        type=str,
                        default=True,
                        help="dertermines quantity of print output")

    args = parser.parse_args()

    verify(
        batch_size=args.batch_size,
        dataset=args.dataset,
        dimension=args.dimension,
        embedding=EMBEDDING,
        #explanations_to_verify_file=args.explanations_to_verify_file,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=args.mode,
        model_file=args.model_file,
        verbose=args.verbose,

        # ComplEx specific hyperparameters
        decay1=args.decay1,
        decay2=args.decay2,
        init=args.init,
        optimizer=args.optimizer,
        reg=args.reg,
    )

if __name__ == "__main__":
    main()