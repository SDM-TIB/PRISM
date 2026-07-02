import argparse
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import ALL_DATASET_NAMES
from src.embeddings.unified_explain import explain
from src.prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER


EMBEDDING = "ConvE"

def main():
    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

    parser.add_argument("--dataset",
                        type=str,
                        default="FR_Reduced_2K",
                        choices=ALL_DATASET_NAMES,
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

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
                        #required=True,
                        help="path of the model to explain the predictions of.")

    parser.add_argument("--predictions_to_explain_file",
                        type=str,
                        default=f"{EMBEDDING.lower()}_fr_reduced_2k_tail_42.csv",
                        #required=True,
                        help="path of the file with the facts to explain the predictions of.")

    parser.add_argument("--coverage",
                        type=int,
                        default=10,
                        help="Number of random entities to extract and convert")

    parser.add_argument("--baseline",
                        type=str,
                        default=None,
                        choices=[None,"expath", "k1", "data_poisoning", "criage"],
                        help="attribute to use when we want to use a baseline rather than the Kelpie engine")

    parser.add_argument("--entities_to_convert",
                        type=str,
                        help="path of the file with the entities to convert (only used by baselines)")

    parser.add_argument("--mode",
                        type=str,
                        default="necessary",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    parser.add_argument("--relevance_threshold",
                        type=float,
                        default=None,
                        help="The relevance acceptance threshold to use")

    prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]
    parser.add_argument('--prefilter',
                        choices=prefilters,
                        default=None,
                        help="Prefilter type in {} to use in pre-filtering".format(prefilters))

    parser.add_argument("--prefilter_threshold",
                        type=int,
                        default=20,
                        help="The number of promising training facts to keep after prefiltering")
            
    parser.add_argument("--rules_file",
                        type=str,
                        #default="yago3-10_rules.csv",
                        default="fr_reduced_2k_rules_optimai.csv",
                        #default="fr_reduced_2k_rules.csv",
                        help="The rule file for symbolic filter")

    parser.add_argument("--second_rules_file",
                        type=str,
                        default="FR_editorial_rules.csv",
                        #default=None,
                        help="The rule file for symbolic filter")

    parser.add_argument("--builder",
                        type=str,
                        #default="rules_kelpie_reverse",
                        #default="rules_heuristic_pca",     #TODO: make list of constants
                        #default="rules_kelpie",
                        #default="rules_heuristic_frequency",
                        default="kelpie",
                        help="dertermines which builder and heuristic is used")
        
    parser.add_argument("--verbose",
                        type=str,
                        default=True,
                        help="dertermines quantity of print output")
        
    parser.add_argument("--pca",
                        type=float,
                        default=0.0,
                        help="the minimum pca confidence for rules (threshold=thr)")

    args = parser.parse_args()
    print(args)

    explain(
        baseline=args.baseline,
        batch_size=args.batch_size,
        builder=args.builder,
        coverage=args.coverage,
        dataset_name=args.dataset,
        dimension=args.dimension,
        embedding=EMBEDDING,
        entities_to_convert=args.entities_to_convert,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=args.mode,
        model_file=args.model_file,
        pca_threshold=args.pca,
        predictions_to_explain_file=args.predictions_to_explain_file,
        prefilter=args.prefilter,
        prefilter_threshold=args.prefilter_threshold,
        relevance_threshold=args.relevance_threshold,
        rules_file=args.rules_file,
        second_rules_file=args.second_rules_file,
        verbose=args.verbose,

        # ConvE specific arguments
        decay_rate=args.decay_rate,
        feature_map_dropout=args.feature_map_dropout,
        hidden_dropout=args.hidden_dropout,
        hidden_size=args.hidden_size,
        input_dropout=args.input_dropout,
        label_smoothing=args.label_smoothing
    )


if __name__ == "__main__":
    main()
