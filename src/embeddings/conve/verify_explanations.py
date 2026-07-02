import argparse
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import ALL_DATASET_NAMES
from src.embeddings.unified_verify_explanations import verify

EMBEDDING = "ConvE"

def main():
    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

    parser.add_argument("--dataset",
                        type=str,
                        default="FR_Reduced_2K",
                        #default="YAGO3-10",
                        choices=ALL_DATASET_NAMES,
                        help=f"The dataset to use: {ALL_DATASET_NAMES}")

    parser.add_argument("--max_epochs",
                        type=int, default=1000,
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

    parser.add_argument("--model_file",
                        type=str,
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
                        #default=f"{EMBEDDING}_YAGO3-10.pt",
                        help="Path where to find the model to explain")

    parser.add_argument("--mode",
                        type=str,
                        default="sufficient",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    # parser.add_argument("--explanations_to_verify_file",
    #                     type=str,
    #                     #default=f"{EMBEDDING}_FR_Reduced_2K_explanation.csv",
    #                     default="TransE_FR_Reduced_2K_explanation.csv",
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

        # ConvE specific hyperparameters
        decay_rate=args.decay_rate,
        feature_map_dropout=args.feature_map_dropout,
        hidden_dropout=args.hidden_dropout,
        hidden_size=args.hidden_size,
        input_dropout=args.input_dropout,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
    