import argparse
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import ALL_DATASET_NAMES
from src.embeddings.unified_explain import explain
from src.prefilters.prefilter import SYMBOLIC_PREFILTER, TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER


EMBEDDING = "TransE"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_epochs",
                        type=int,
                        default=200,    #200    #250    #100    #200    #100
                        help="Number of epochs.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=2048,   #2048   #2048   #2048   #2048   #2048
                        help="Batch size.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,  #0.01   #0.01   #0.01   #0.01   #0.01
                        help="Learning rate.")

    parser.add_argument("--dimension",
                        type=int,
                        default=200,#200,    #200    #50     #50     #50     #200
                        help="Embedding dimensionality.")

    parser.add_argument("--margin",
                        type=int,
                        default=2,      #2      #2      #5      #2      #5
                        help="Margin for pairwise ranking loss.")

    parser.add_argument("--negative_samples_ratio",
                        type=int,
                        default=5,     #5      #5      #15     #5      #10
                        help="Number of negative samples for each positive sample.")

    parser.add_argument("--regularizer_weight",
                        type=float,
                        default=2.0,   #2.0    #0      #1.0    #50.0   #50.0
                        help="Weight for L2 regularization.")

    parser.add_argument("--coverage",
                        type=int,
                        default=3, #10,
                        help="Number of random entities to extract and convert")

    parser.add_argument("--baseline",
                        type=str,
                        default=None,
                        choices=[None, "k1", "data_poisoning", "criage"],
                        help="attribute to use when we want to use a baseline rather than the Kelpie engine")

    parser.add_argument("--entities_to_convert",
                        type=str,
                        help="path of the file with the entities to convert (only used by baselines)")

    parser.add_argument("--mode",
                        type=str,
                        default="sufficient",
                        choices=["sufficient", "necessary"],
                        help="The explanation mode")

    parser.add_argument("--relevance_threshold",
                        type=float,
                        default=None,
                        help="The relevance acceptance threshold to use. \nDefault for stochasticNecessaryExplanationBuilder=5, for stochasticSufficientExplanationBuilder=0.9")

    parser.add_argument("--prefilter_threshold",
                        type=int,
                        default=20,
                        help="The number of promising training facts to keep after prefiltering")

    prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER, SYMBOLIC_PREFILTER]
    parser.add_argument('--prefilter',
                        choices=prefilters,
                        default=None,
                        help="Prefilter type in {} to use in pre-filtering".format(prefilters))
    
    parser.add_argument("--verbose",
                        type=str,
                        default=True,
                        help="dertermines quantity of print output")

    parser.add_argument("--dataset",    #FB15k  #WN18   #FB15   #WN18RR #YAGO3-10
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        default="FR_Reduced_2K",
                        #default="YAGO3-10",
                        help=f"The dataset to use: {ALL_DATASET_NAMES}")

    parser.add_argument("--model_file",
                        type=str,
                        #default=f"{EMBEDDING}_YAGO3-10.pt",
                        default=f"{EMBEDDING}_FR_Reduced_2K.pt",
                        help="Path where to find the model to explain")

    parser.add_argument("--predictions_to_explain_file",
                        type=str,
                        #default=default=f"{EMBEDDING.lower()}__FR_Lutgardis.csv",
                        #default=f"{EMBEDDING.lower()}_fr_reduced_2k_tail_0.csv",
                        #default=f"{EMBEDDING.lower()}_fr_reduced_2k_tail_42.csv",
                        default=f"FR_10.csv",
                        help="Path where to find the facts to explain")
    
    parser.add_argument("--rules_file",
                        type=str,
                        #default="yago3-10_rules.csv",
                        default="fr_reduced_2k_rules.csv",
                        help="The rule file for symbolic filter")
    
    parser.add_argument("--second_rules_file",
                        type=str,
                        default="FR_editorial_rules.csv",
                        help="The rule file for symbolic filter")
    
    parser.add_argument("--builder",
                        type=str,
                        #default="rules_kelpie_reverse",
                        default="rules_heuristic_pca_test",
                        #default="rules_kelpie",
                        #default="rules_heuristic_frequency",
                        #default="kelpie",
                        help="dertermines which builder and heuristic is used")
    
    parser.add_argument("--pca",
                        type=float,
                        default=0.7,
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

        # TransE specific hyperparameters
        margin=args.margin,
        negative_samples_ratio=args.negative_samples_ratio,
        regularizer_weight=args.regularizer_weight
        )


if __name__ == "__main__":
    main()
