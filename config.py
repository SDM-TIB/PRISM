import os

ROOT = os.path.realpath(os.path.join(os.path.abspath(__file__), ".."))
RULES_PATH = os.path.join(ROOT, "data", "0_rules")
DATA_PATH = os.path.join(ROOT, "data", "1_triple_store")
MODEL_PATH = os.path.join(ROOT, "data", "2_stored_models")
RANKS_PATH = os.path.join(ROOT, "data", "3_filtered_ranks")
PREDICTIONS_PATH = os.path.join(ROOT, "data", "4_predictions")
EXPLAIN_PATH = os.path.join(ROOT, "data", "5_explanations")
STATISTICS_PATH = os.path.join(ROOT, "data", "6_statistics")
LOGGING_PATH = os.path.join(ROOT, "data", "7_logging")
MAX_PROCESSES = 8

FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10SPLAIN = "YAGO3-10SPLAIN"
YAGO3_10 = "YAGO3-10"
FR_ADELA_3K = "FR_ADELA_3K"
FR_EP_13K = "FR_EP_13K"
FR_FULL_12K = "FR_FULL_12K"
FR_Reduced_2K="FR_Reduced_2K"
FR_Reduced_2K_copy="FR_Reduced_2K_copy"
DB100K ="DB100K"

ALL_DATASET_NAMES = [DB100K, FB15K, YAGO3_10SPLAIN, FB15K_237, WN18, WN18RR, YAGO3_10, FR_ADELA_3K, FR_EP_13K, FR_FULL_12K, FR_Reduced_2K, FR_Reduced_2K_copy]