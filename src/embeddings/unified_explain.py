import numpy
import os
import random
import sys
import time
import torch


sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from config import EXPLAIN_PATH, PREDICTIONS_PATH, MODEL_PATH, LOGGING_PATH, RULES_PATH, STATISTICS_PATH
from src.link_prediction.models.transe import TransE
from src.link_prediction.models.complex import ComplEx
from src.link_prediction.models.conve import ConvE
from src.link_prediction.models.model import DIMENSION, LEARNING_RATE, EPOCHS, BATCH_SIZE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, INIT_SCALE, OPTIMIZER_NAME, DECAY_1, DECAY_2, REGULARIZER_WEIGHT, REGULARIZER_NAME, MARGIN, NEGATIVE_SAMPLES_RATIO
from src.model.utils import arguments, generate_output_file_prefix, concat_output
from src.model.criage import Criage
from src.model.eXpath import eXpath
from src.model.data_poisoning import DataPoisoning
from src.model.dataset import Dataset
from src.model.kelpie import Kelpie as Kelpie
from src.model.ruleset import EmptyResultException, Ruleset


def explain(
    baseline: str,
    batch_size: int,
    builder: str,
    coverage: int,
    dataset_name: str,
    dimension: int,
    embedding: str,
    entities_to_convert: str,
    learning_rate: float,
    max_epochs: int,
    mode: str,
    model_file: str,
    pca_threshold: float,
    predictions_to_explain_file: str,
    prefilter: str,
    prefilter_threshold: int,
    relevance_threshold: float,
    rules_file: str,
    second_rules_file: str,
    verbose: bool,

    # TransE specific hyperparameters
    margin: int = 5,
    negative_samples_ratio: int = 10,
    regularizer_weight: float = 50.0,

    # ConvE specific hyperparameters
    decay_rate: float = 1.0,
    feature_map_dropout: float = 0.5,
    hidden_dropout: float = 0.4,
    hidden_size: int = 9728,
    input_dropout: float = 0.3,
    label_smoothing: float = 0.1,

    # ComplEx specific hyperparameters
    optimizer:str = "Adagrad",
    reg:float = 5e-3,
    init:float = 1e-3,
    decay1:float = 0.9,
    decay2:float = 0.999
) -> dict:
    """
    Explain the model with the given parameters.

    Args:
    dataset (str): Name of the dataset to use.
    max_epochs (int): Number of epochs.
    batch_size (int): Batch size.
    learning_rate (float): Learning rate.
    dimension (int): Embedding dimensionality.
    margin (int): Margin for pairwise ranking loss.
    negative_samples_ratio (int): Number of negative samples for each positive sample.
    regularizer_weight (float): Weight for L2 regularization.
    model_file (str): Path where to find the model to explain.
    predictions_to_explain_file (str): Path where to find the facts to explain.
    regularizer_weight (float): Weight for L2 regularization. Number of random entities to extract and convert.
    baseline (str, optional): Baseline attribute. Defaults to None.
    entities_to_convert (str, optional): Path of the file with the entities to convert.
    mode (str): The explanation mode.
    relevance_threshold (float, optional): The relevance acceptance threshold to use.
    prefilter (str): Prefilter type to use in pre-filtering.
    prefilter_threshold (int): The number of promising training facts to keep after prefiltering.
    rules_file (str): The rule file for symbolic filter.
    second_rules_file (str): The second rule file for symbolic filter.
    verbose (bool): Determines quantity of print output.
    builder (str): Determines which builder and heuristic is used.
    """
    
    # embedding specific hyperparameters
    transe_hyperparameters = {DIMENSION: dimension,
                BATCH_SIZE: batch_size,
                LEARNING_RATE: learning_rate,
                EPOCHS: max_epochs,
                MARGIN: margin,
                NEGATIVE_SAMPLES_RATIO: negative_samples_ratio,
                REGULARIZER_WEIGHT: regularizer_weight}

    conve_hyperparameters = {DIMENSION: dimension,
                BATCH_SIZE: batch_size,
                LEARNING_RATE: learning_rate,
                EPOCHS: max_epochs,
                INPUT_DROPOUT: input_dropout,
                FEATURE_MAP_DROPOUT: feature_map_dropout,
                HIDDEN_DROPOUT: hidden_dropout,
                HIDDEN_LAYER_SIZE: hidden_size,
                DECAY: decay_rate,
                LABEL_SMOOTHING: label_smoothing}
    
    complex_hyperparameters = {DIMENSION: dimension,
                INIT_SCALE: init,
                LEARNING_RATE: learning_rate,
                OPTIMIZER_NAME: optimizer,
                DECAY_1: decay1,
                DECAY_2: decay2,
                REGULARIZER_WEIGHT: reg,
                EPOCHS: max_epochs,
                BATCH_SIZE: batch_size,
                REGULARIZER_NAME: "N3"}

    # deterministic!
    seed = 42
    torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    kernel = ""
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        kernel = "cuda"
    else:
        torch.set_rng_state(torch.get_rng_state())
        kernel = "cpu"

    if prefilter == None:
        if any(keyword in builder.lower() for keyword in ["rules", "heuristic", "pca", "frequency", "reverse"]):
            prefilter = "symbolic_based"
        elif ("kelpie" in builder.lower()):
            prefilter = "topology_based"
        else:
            raise Exception("no valid builder on which to choose prefilter: "+ builder)

    args_dict = arguments()

    # select model and hyperparameters
    embeddings_map = {
        "transe": (TransE, transe_hyperparameters),
        "conve": (ConvE, conve_hyperparameters),
        "complex": (ComplEx, complex_hyperparameters)
    }
    model_type, hyperparameters = embeddings_map[embedding.lower()]

    model_path = os.path.join(MODEL_PATH, model_file)
    predictions_to_explain_path =  os.path.join(PREDICTIONS_PATH, predictions_to_explain_file)
    second_rules_path = os.path.join(RULES_PATH, second_rules_file) \
        if second_rules_file != None else None

    ########## LOAD DATASET & RULESSET

    # load the dataset and its training samples
    print("Loading dataset %s..." % dataset_name)
    dataset = Dataset(name=dataset_name, separator="\t", load=True)
    if prefilter == "symbolic_based":
        ruleset = Ruleset(dataset, os.path.join(RULES_PATH, rules_file), second_rules_path, verbose=verbose, pca_threshold=pca_threshold)
    else:
        ruleset=None

    output_prefix, timestamp = generate_output_file_prefix(embedding=embedding)
    filename_8_explain = output_prefix + "_08_explain_py.txt"
    print(filename_8_explain)
    filename_9_output = output_prefix + f"_09_{dataset.name}_output.csv"
    print(filename_9_output)
    with open(filename_8_explain, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"explain.py at {timestamp}\n")
        execution_log.write(str(args_dict)+"\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, output: {filename_9_output}\n")

    print(f"Reading predictions to explain... {predictions_to_explain_path}")
    with open(predictions_to_explain_path, "r", encoding="utf8" ) as facts_file:
        testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

    model = model_type(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    
    if torch.cuda.is_available():
        model.to('cuda')
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
    model.eval()

    output_lines = []

    global_statistics = {"date":timestamp, "facts":0, "complexity_list":[], "time_list":[], 'empty':[]} 

    ###### start time Globally ######
    start_time_overhead = time.time()

    if baseline is None:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, output_prefix=output_prefix, ruleset=ruleset,
                        verbose=verbose, builder=builder)
    elif baseline == "data_poisoning":
        kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter, output_prefix=output_prefix)
    elif baseline == "criage":
        kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters,output_prefix=output_prefix)
    elif baseline == "k1":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, output_prefix=output_prefix, max_explanation_length=1)
    elif baseline == "expath":
         kelpie = eXpath(model=model, dataset=dataset, hyperparameters=hyperparameters, relevance_threshold=relevance_threshold, max_explanation_length=1)
    else:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, output_prefix=output_prefix, ruleset=ruleset,
                        verbose=verbose, builder=builder)

    testing_fact_2_entities_to_convert = None
    if mode == "sufficient" and entities_to_convert is not None:
        print("Reading entities to convert...")
        testing_fact_2_entities_to_convert = {}
        with open(entities_to_convert, "r", encoding="utf8" ) as entities_to_convert_file:
            entities_to_convert_lines = entities_to_convert_file.readlines()
            i = 0
            while i < len(entities_to_convert_lines):
                cur_head, cur_rel, cur_name = entities_to_convert_lines[i].strip().split(";")
                assert [cur_head, cur_rel, cur_name] in testing_facts
                cur_entities_to_convert = entities_to_convert_lines[i + 1].strip().split(",")
                testing_fact_2_entities_to_convert[(cur_head, cur_rel, cur_name)] = cur_entities_to_convert
                i += 3

    end_time_overhead = time.time()
    duration_overhead = end_time_overhead - start_time_overhead

    for i, fact in enumerate(testing_facts):
        ###### start time per sample ######
        start_time_loop = time.time()
        
        try:
            head, relation, tail = fact
        except:                             # if row is empty or mal formatted
            if verbose: print(f"skipped explaining row '{', '.join(fact)}', prediction might be malformed or empty")   # print(*fact, sep = ", ")
            continue
        print("\nExplaining fact " + str(i) + " of " + str(
            len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
        head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                        dataset.get_id_for_relation_name(relation), \
                                        dataset.get_id_for_entity_name(tail)
        sample_to_explain = (head_id, relation_id, tail_id)

        if mode == "sufficient":
            try:
                entities_to_convert_ids = None if testing_fact_2_entities_to_convert is None \
                    else [dataset.entity_name_2_id[x] for x in testing_fact_2_entities_to_convert[(head, relation, tail)]]
                
                rule_samples_with_relevance, entities_to_convert_ids, complexity = \
                    kelpie.explain_sufficient(
                        sample_to_explain=sample_to_explain,
                        perspective="head",
                        num_promising_samples=prefilter_threshold,
                        num_entities_to_convert=coverage,
                        entities_to_convert=entities_to_convert_ids)
                
                print(complexity)

                if entities_to_convert_ids is None or len(entities_to_convert_ids) == 0:
                    continue
                entities_to_convert = [dataset.entity_id_2_name[x] for x in entities_to_convert_ids]

                global_statistics["complexity_list"] += [sum(complexity.values())]
                global_statistics["facts"] += 1

            except EmptyResultException as e:
                print(fact)
                print(e)
                global_statistics['empty'].append((sample_to_explain,fact,e)) #TODO: remove
                continue
            
            output_lines += concat_output(
                rule_samples_with_relevance, 
                dataset, fact, verbose, entities_to_convert=entities_to_convert)

        elif mode == "necessary":
            try:
                rule_samples_with_relevance, complexity = \
                    kelpie.explain_necessary(
                        sample_to_explain=sample_to_explain,
                        perspective="head",
                        num_promising_samples=prefilter_threshold)
                
                print(complexity)
                
                global_statistics["complexity_list"] += [sum(complexity.values())]
                global_statistics["facts"] += 1

            except EmptyResultException as e:
                print(fact)
                print(e)
                global_statistics['empty'].append((sample_to_explain,fact,e)) #TODO: remove
                continue

            output_lines += concat_output(
                rule_samples_with_relevance,
                dataset, fact, verbose)

        if 0 in complexity: 
            print(f"complexity of approach: {sum(complexity.values())} retrainings, {complexity[0]} for preliminary relevance")
        elif bool(complexity): 
            print(f"complexity of approach: {sum(complexity.values())} retrainings")

        end_time_loop = time.time()
        duration = end_time_loop - start_time_loop
        global_statistics["time_list"] += [duration]

    try:
        #print("Required time: " + str(duration) + " seconds")
        global_statistics["algorithm"] = builder
        global_statistics["predictions"] = predictions_to_explain_file[:-4]
        partial_overhead = duration_overhead/global_statistics["facts"]
        global_statistics["time_list"] = [(sampletime + partial_overhead) for sampletime in global_statistics["time_list"]]
        global_statistics["time_total"] = round(sum(global_statistics["time_list"])+duration_overhead, 1)
        global_statistics["mean_time"] = round(numpy.mean(global_statistics["time_list"]), 1)
        global_statistics["median_time"] = round(numpy.median(global_statistics["time_list"]), 1)
        global_statistics["mean_complexity"] = round(numpy.mean(global_statistics["complexity_list"]), 1)
        global_statistics["median_complexity"] = round(numpy.median(global_statistics["complexity_list"]), 1)

        print(filename_9_output)
        with open(os.path.join(LOGGING_PATH, filename_9_output), "w", encoding="utf8" ) as output:
            output.writelines(output_lines)
        
        filename_original_output = os.path.join(EXPLAIN_PATH, f"{model_file[:-3]}_explanation.csv") # caution, when this is changed, the inputfile in the verify code needs to be changed too
        with open(filename_original_output, "w", encoding="utf8" ) as original_output:
            original_output.writelines(output_lines)
        
        print(filename_8_explain)
        with open(os.path.join(LOGGING_PATH, filename_8_explain), "a", encoding="utf8" ) as output:
            output.writelines(str(global_statistics))

        filename_explain_output = os.path.join(STATISTICS_PATH, f"{dataset_name}_statistics.csv")
        print(filename_explain_output)
        if not os.path.isfile(filename_explain_output):
            with open(filename_explain_output, "w", encoding="utf8" ) as output:
                output.writelines(";".join([str(val) for val in global_statistics.keys()])+"\n")
        with open(filename_explain_output, "a", encoding="utf8" ) as output:
            output.writelines(";".join([str(val) for val in global_statistics.values()])+"\n")

    except Exception as e:
        print("output_prefix: ", output_prefix) 
        print("timestamp: ", timestamp) 
        print("statistics: ", global_statistics)
        print(e)

    print(global_statistics)
    return global_statistics
