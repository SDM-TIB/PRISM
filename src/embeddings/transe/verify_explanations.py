import argparse
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from src.embeddings.unified_verify_explanations import verify
from config import ALL_DATASET_NAMES

EMBEDDING = "TransE"

def main():

    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")


    parser.add_argument("--max_epochs",
                        type=int,
                        default=200,
                        help="Number of epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=2048,
                        help="Batch size.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,
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
                        default=3,
                        help="Number of negative samples for each positive sample.")

    parser.add_argument("--regularizer_weight",
                        type=float,
                        default=50.0,
                        help="Weight for L2 regularization.")

    parser.add_argument("--mode",
                        type=str,
                        default="sufficient",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    parser.add_argument("--dataset",
                        type=str,
                        default="FR_Reduced_2K",
                        #default="YAGO3-10",
                        choices=ALL_DATASET_NAMES,
                        help=f"The dataset to use: {ALL_DATASET_NAMES}")

    parser.add_argument("--model_file",
                        type=str,
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
                        #default=f"{EMBEDDING}_YAGO3-10.pt",
                        help="Path where to find the model to explain")

    # parser.add_argument("--explanations_to_verify_file",
    #                     type=str,
    #                     default=f"{EMBEDDING}_FR_Reduced_2K_explanation.csv",
    #                     #default=f"{EMBEDDING}_YAGO3-10_explanation.csv",
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

        # TransE specific hyperparameters
        margin=args.margin,
        negative_samples_ratio=args.negative_samples_ratio,
        regularizer_weight=args.regularizer_weight,
    )

if __name__ == "__main__":
    main()