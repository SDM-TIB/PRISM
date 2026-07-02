import argparse
import copy
import numpy
import os
import pandas as pd
import random
import sys
import torch


sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import EXPLAIN_PATH, MODEL_PATH, LOGGING_PATH, STATISTICS_PATH
from src.model.utils import arguments, flatten_list, seperate_rule_triples, generate_output_file_prefix
from src.model.dataset import Dataset, MANY_TO_ONE, ONE_TO_ONE
from src.link_prediction.optimization.bce_optimizer import BCEOptimizer
from src.link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from src.link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from src.link_prediction.models.transe import TransE
from src.link_prediction.models.complex import ComplEx
from src.link_prediction.models.conve import ConvE
from src.link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, OPTIMIZER_NAME, DECAY_1, DECAY_2, INIT_SCALE, REGULARIZER_NAME


def verify(
    batch_size: int,
    dataset: str,
    dimension: int,
    embedding: str,
    learning_rate: float,
    max_epochs: int,
    mode: str,
    model_file: str,
    verbose: bool,
    
    # TransE specific hyperparameters
    margin: int = 5,
    negative_samples_ratio: int = 10,
    regularizer_weight: float = 50.0,

    # ConvE specific hyperparameters
    decay_rate:float = 0.1,
    feature_map_dropout:float = 0.3,
    hidden_dropout:float = 0.4,
    hidden_size:int = 9728,
    input_dropout:float = 0.3,
    label_smoothing:float = 0.1,

    # ComplEx specific hyperparameters
    decay1:float = 0.9,
    decay2:float = 0.999,
    init:float = 1e-3,
    optimizer:str = 'Adagrad',
    reg:float = 0.0 
):
    
    # embedding specific hyperparameters
    transe_hyperparameters = {DIMENSION: dimension,
                    EPOCHS: max_epochs,
                    BATCH_SIZE: batch_size,
                    LEARNING_RATE: learning_rate,
                    MARGIN: margin,
                    NEGATIVE_SAMPLES_RATIO: negative_samples_ratio,
                    REGULARIZER_WEIGHT: regularizer_weight}
    
    conve_hyperparameters = {DIMENSION: dimension,
                   EPOCHS: max_epochs,
                   BATCH_SIZE: batch_size,
                   LEARNING_RATE: learning_rate,
                   INPUT_DROPOUT: input_dropout,
                   FEATURE_MAP_DROPOUT: feature_map_dropout,
                   HIDDEN_DROPOUT: hidden_dropout,
                   HIDDEN_LAYER_SIZE: hidden_size,
                   DECAY: decay_rate,
                   LABEL_SMOOTHING: label_smoothing}
    
    complex_hyperparameters = {DIMENSION: dimension,
                EPOCHS: max_epochs,
                BATCH_SIZE: batch_size,
                LEARNING_RATE: learning_rate,
                INIT_SCALE: init,
                OPTIMIZER_NAME: optimizer,
                DECAY_1: decay1,
                DECAY_2: decay2,
                REGULARIZER_NAME: "N3",
                REGULARIZER_WEIGHT: reg}

    # select model and hyperparameters
    embeddings_map = {
        "transe": (TransE, transe_hyperparameters, PairwiseRankingOptimizer),
        "conve": (ConvE, conve_hyperparameters, BCEOptimizer),
        "complex": (ComplEx, complex_hyperparameters, MultiClassNLLOptimizer)
    }
    emb_model, hyperparameters, emb_optimizer = embeddings_map[embedding.lower()]

    model_path = os.path.join(MODEL_PATH, model_file)

    explanations_to_verify_file = f"{embedding}_{dataset}_explanation.csv"  #"TransE_FR_Reduced_2K_explanation.csv"
    explanations_path = os.path.join(EXPLAIN_PATH, explanations_to_verify_file)
    
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

    args_dict = arguments()

    statistics = {"samples": 0, "original_direct_score": 0, "new_direct_score": 0, "original_tail_rank": 0, "new_tail_rank": 0, "rank_change_list": [], "score_change_list": []}
    
    ########## LOAD DATASET
    # load the dataset and its training samples
    print("Loading dataset %s..." % dataset)
    dataset = Dataset(name=dataset, separator="\t", load=True)

    output_prefix, timestamp = generate_output_file_prefix(embedding=embedding)
    log_path = os.path.join(LOGGING_PATH, output_prefix + "_11_verify_py.txt")
    print(log_path)
    log_end_to_end_output_path = os.path.join(LOGGING_PATH, output_prefix + f"_12_{dataset.name}_output_end_to_end.txt")
    print(log_end_to_end_output_path)

    with open(log_path, "w", encoding="utf8") as execution_log:
        execution_log.write(f"verify_explanations.py at {timestamp}\n")
        execution_log.write(str(args_dict) + "\n")
        execution_log.write(f"kernel: {kernel}, seed: {seed}, output: {log_end_to_end_output_path}\n")

    with open(explanations_path, "r", encoding="utf8") as input_file:
        expl_to_verify = input_file.readlines()

    def predict_samples_in_batches(model_to_predict_batches, samples_to_predict_in_batches, batch_size=20):
        scores, ranks, predictions = [], [], []

        start = 0
        while start < len(samples_to_predict_in_batches):
            end = min(len(samples_to_predict_in_batches), start + batch_size)

            cur_batch = numpy.array(samples_to_predict_in_batches[start:end])
            cur_batch_scores, cur_batch_ranks, cur_batch_predictions = model_to_predict_batches.predict_samples(cur_batch)
            scores += cur_batch_scores
            ranks += cur_batch_ranks
            predictions += cur_batch_predictions

            start += batch_size

        return scores, ranks, predictions

    original_model = emb_model(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    
    if kernel == "cuda":
        original_model.to(kernel)
        original_model.load_state_dict(torch.load(model_path))
    elif kernel == "cpu":
        original_model.load_state_dict(torch.load(model_path, map_location=torch.device(kernel)))
    original_model.eval()

    facts_to_explain = []
    samples_to_explain = []
    perspective = "head"    # for all samples the perspective was head for simplicity
    sample_to_explain_2_best_rule = {}

    if mode == "sufficient":

        sample_to_explain_2_entities_to_convert = {}

        i = 0
        while i <= len(expl_to_verify)-4:
            fact_line = expl_to_verify[i]
            similar_entities_line = expl_to_verify[i+1]
            rules_line = expl_to_verify[i+2]
            empty_line = expl_to_verify[i+3]

            # sample to explain
            fact = tuple(fact_line.strip().split(";"))
            facts_to_explain.append(fact)
            sample = (dataset.entity_name_2_id[fact[0].lower()], dataset.relation_name_2_id[fact[1].lower()], dataset.entity_name_2_id[fact[2].lower()])
            samples_to_explain.append(sample)

            # similar entities
            try:
                similar_entities_names = similar_entities_line.strip().split(",")
            except:
                raise ValueError(f"malformed explanations file. Make sure that the file is formatted correctly for {mode} explanations. \n")
            similar_entities = [dataset.entity_name_2_id[x.strip()] for x in similar_entities_names]
            sample_to_explain_2_entities_to_convert[sample] = similar_entities

            # rules
            rules_with_relevance = []

            try:
                rule_relevance_inputs = rules_line.strip().split("|")
            except:
                raise ValueError(f"malformed explanations file. Make sure that the file is formatted correctly for {mode} explanations. \n")
            best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split(":")
            best_rule_list = best_rule.strip().split(";")
            best_triples = [rule.strip().split(",") for rule in best_rule_list]
            
            length = [len(item) for item in best_triples]
            if max(length) > 3:
                best_triples = seperate_rule_triples(best_triples)

            best_rule_samples = [dataset.fact_to_sample(x) for x in best_triples]

            relevance = float(best_rule_relevance_str)
            rules_with_relevance.append((best_rule_samples, relevance))

            sample_to_explain_2_best_rule[sample] = best_rule_samples
            i += 4  # sufficient explanation files have 3 rows + 1 blanc 

        samples_to_add = []         # the samples to add to the training set before retraining
        samples_to_convert = []     # the samples that, after retraining, should have changed their predictions

        # for each sample to explain, get the corresponding similar entities and the most relevant sample in addition.
        # For each of those similar entities create
        #   - a version of the sample to explain that features the similar entity instead of the entity to explain
        #   - a version of the most relevant sample to add that features the similar entity instead of the entity to explain

        sample_to_convert_2_original_sample_to_explain = {}
        samples_to_convert_2_added_samples = {}
        for sample_to_explain in samples_to_explain:

            entity_to_explain = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]

            cur_entities_to_convert = sample_to_explain_2_entities_to_convert[sample_to_explain]

            cur_best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]

            for cur_entity_to_convert in cur_entities_to_convert:
                cur_sample_to_convert = Dataset.replace_entity_in_sample(sample=sample_to_explain,
                                                                        old_entity=entity_to_explain,
                                                                        new_entity=cur_entity_to_convert,
                                                                        as_numpy=False)
                cur_samples_to_add = Dataset.replace_entity_in_samples(samples=cur_best_rule_samples,
                                                                    old_entity=entity_to_explain,
                                                                    new_entity=cur_entity_to_convert,
                                                                    as_numpy=False)

                samples_to_convert.append(cur_sample_to_convert)
                samples_to_convert_2_added_samples[cur_sample_to_convert] = cur_samples_to_add

                for cur_sample_to_add in cur_samples_to_add:
                    samples_to_add.append(cur_sample_to_add)

                sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)] = sample_to_explain

        new_dataset = copy.deepcopy(dataset)


        # if any of the samples_to_add overlaps contradicts any pre-existing facts
        # (e.g. adding "<Obama, born_in, Paris>" when the dataset already contains "<Obama, born_in, Honolulu>")
        # we need to remove such pre-eisting facts before adding the new samples_to_add
        print("Adding samples: ")
        for (head, relation, tail) in samples_to_add:
            print("\t" + dataset.printable_sample((head, relation, tail)))
            if new_dataset.relation_2_type[relation] in [MANY_TO_ONE, ONE_TO_ONE]:
                for pre_existing_tail in new_dataset.to_filter[(head, relation)]:
                    new_dataset.remove_training_sample(numpy.array((head, relation, pre_existing_tail)))

        # append the samples_to_add to training samples of new_dataset
        # (and also update new_dataset.to_filter accordingly)
        new_dataset.add_training_samples(numpy.array(samples_to_add))

        # obtain tail ranks and scores of the original model for that all_samples_to_convert

        original_scores, original_ranks, original_predictions = predict_samples_in_batches(original_model, samples_to_convert)

        new_model = emb_model(dataset=new_dataset,
                        hyperparameters=hyperparameters,
                        init_random=True)
        new_optimizer = emb_optimizer(model=new_model, hyperparameters=hyperparameters, verbose=verbose)
        new_optimizer.train(train_samples=new_dataset.train_samples)
        new_model.eval()

        new_scores, new_ranks, new_predictions = predict_samples_in_batches(new_model, samples_to_convert)

        for i in range(len(samples_to_convert)):
            cur_sample = samples_to_convert[i]
            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                                dataset.relation_id_2_name[cur_sample[1]],
                                dataset.entity_id_2_name[cur_sample[2]]]) + ">")
            print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
            print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
            print()

        cumulated_statisticss = {}
        output_lines = []
        for i in range(len(samples_to_convert)):
            cur_sample_to_convert = samples_to_convert[i]
            cur_added_samples = samples_to_add[i]
            original_sample_to_explain = sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)]

            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            a = ";".join(dataset.sample_to_fact(original_sample_to_explain))
            b = ";".join(dataset.sample_to_fact(cur_sample_to_convert))

            c = []
            samples_to_add_to_this_entity = samples_to_convert_2_added_samples[cur_sample_to_convert]
            for x in range(4):
                if x < len(samples_to_add_to_this_entity):
                    c.append(";".join(dataset.sample_to_fact(samples_to_add_to_this_entity[x])))
                else:
                    c.append(";;")

            c = ";".join(c)
            d = str(original_direct_score) + ";" + str(new_direct_score)
            e = str(original_tail_rank) + ";" + str(new_tail_rank)
            output_lines.append(";".join([a, b, c, d, e]) + "\n")

            # there are as many samples per original as the coverage was set.
            # for explain with coverage 3 -> 3 samples per original
            # for 10 samples to explain -> 30 samples in total 
            if original_sample_to_explain not in cumulated_statisticss:
                cumulated_statisticss[original_sample_to_explain] = {"original_direct_score": 0, "new_direct_score": 0, "original_tail_rank": 0, "new_tail_rank": 0, "rank_change_list": [], "score_change_list": [], "samples": 0}
            cumulated_statisticss[original_sample_to_explain]["original_direct_score"] \
                += original_direct_score
            cumulated_statisticss[original_sample_to_explain]["new_direct_score"] \
                += new_direct_score
            cumulated_statisticss[original_sample_to_explain]["original_tail_rank"] \
                += original_tail_rank
            cumulated_statisticss[original_sample_to_explain]["new_tail_rank"] \
                += new_tail_rank
            cumulated_statisticss[original_sample_to_explain]["samples"] += 1 
            
        # calculate avg for all the accumulated values
        for key in cumulated_statisticss:
            cumulated_statisticss[key]["original_direct_score"] \
                /= cumulated_statisticss[key]["samples"]
            cumulated_statisticss[key]["new_direct_score"] \
                /= cumulated_statisticss[key]["samples"]
            cumulated_statisticss[key]["original_tail_rank"] \
                /= cumulated_statisticss[key]["samples"]
            cumulated_statisticss[key]["new_tail_rank"] \
                /= cumulated_statisticss[key]["samples"]
            
            cumulated_statisticss[key]["score_change_list"] \
                += [cumulated_statisticss[key]["new_direct_score"] 
                    - cumulated_statisticss[key]["original_direct_score"]]
            cumulated_statisticss[key]["rank_change_list"] \
                += [cumulated_statisticss[key]["new_tail_rank"] 
                    - cumulated_statisticss[key]["original_tail_rank"]]

        statistics["samples"] = len(cumulated_statisticss)
        statistics["original_direct_score"] = sum([cumulated_statisticss[key]["original_direct_score"] for key in cumulated_statisticss])
        statistics["new_direct_score"] = sum([cumulated_statisticss[key]["new_direct_score"] for key in cumulated_statisticss])
        statistics["original_tail_rank"] = sum([cumulated_statisticss[key]["original_tail_rank"] for key in cumulated_statisticss])
        statistics["new_tail_rank"] = sum([cumulated_statisticss[key]["new_tail_rank"] for key in cumulated_statisticss])
        statistics["score_change_list"] = flatten_list([cumulated_statisticss[key]["score_change_list"] for key in cumulated_statisticss])
        statistics["rank_change_list"] = flatten_list([cumulated_statisticss[key]["rank_change_list"] for key in cumulated_statisticss])

        print(log_end_to_end_output_path)
        with open(log_end_to_end_output_path, "w", encoding="utf8" ) as outfile:
            outfile.writelines(output_lines)

    elif mode == "necessary":
        i = 0
        while i <= len(expl_to_verify) -3:
            fact_line = expl_to_verify[i]
            rules_line = expl_to_verify[i + 1]
            empty_line = expl_to_verify[i + 2]

            # if unexplained, skip
            if len(rules_line) < 3:
                print(f"skipping: '{fact_line}' because rules_line too small: '{rules_line}'")
                i += 3
                continue

            # sample to explain
            fact = tuple(fact_line.strip().split(";"))
            facts_to_explain.append(fact)
            sample = (dataset.entity_name_2_id[fact[0].lower()], dataset.relation_name_2_id[fact[1].lower()], dataset.entity_name_2_id[fact[2].lower()])
            samples_to_explain.append(sample)

            # rules
            rules_with_relevance = []

            try:
                rule_relevance_inputs = rules_line.strip().split("|")
            except:
                raise ValueError(f"malformed explanations file. Make sure that the file is formatted correctly for {mode} explanations. \n")
            best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split(":")
            best_rule_list = best_rule.strip().split(";")
            best_triples = [rule.strip().split(",") for rule in best_rule_list]
            
            length = [len(item) for item in best_triples]
            if max(length) > 3:
                best_triples = seperate_rule_triples(best_triples)

            best_rule_samples = [dataset.fact_to_sample(x) for x in best_triples]
            relevance = float(best_rule_relevance_str)
            rules_with_relevance.append((best_rule_samples, relevance))

            sample_to_explain_2_best_rule[sample] = best_rule_samples
            i += 3  # necessary explanation files have 2 rows + 1 blanc 

        samples_to_remove = []  # the samples to remove from the training set before retraining

        for sample_to_explain in samples_to_explain:
            best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
            samples_to_remove += best_rule_samples

        new_dataset = copy.deepcopy(dataset)

        print("Removing samples: ")
        for (head, relation, tail) in samples_to_remove:
            print("\t" + dataset.printable_sample((head, relation, tail)))

        # remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
        new_dataset.remove_training_samples(numpy.array(samples_to_remove))

        # obtain tail ranks and scores of the original model for all samples_to_explain
        original_scores, original_ranks, original_predictions = predict_samples_in_batches(original_model, samples_to_explain)

        ######

        new_model = emb_model(dataset=new_dataset,
                        hyperparameters=hyperparameters,
                        init_random=True) 
        new_optimizer = emb_optimizer(model=new_model, hyperparameters=hyperparameters, verbose=verbose)
        new_optimizer.train(train_samples=new_dataset.train_samples)
        new_model.eval()

        #new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))
        new_scores, new_ranks, new_predictions = predict_samples_in_batches(new_model, samples_to_explain)

        for i in range(len(samples_to_explain)):
            cur_sample = samples_to_explain[i]
            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                                dataset.relation_id_2_name[cur_sample[1]],
                                dataset.entity_id_2_name[cur_sample[2]]]) + ">")
            print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
            print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
            print()

        output_lines = []
        for i in range(len(samples_to_explain)):
            cur_sample_to_explain = samples_to_explain[i]

            original_direct_score = original_scores[i][0]
            original_tail_rank = original_ranks[i][1]

            new_direct_score = new_scores[i][0]
            new_tail_rank = new_ranks[i][1]

            a = ";".join(dataset.sample_to_fact(cur_sample_to_explain))

            b = []
            samples_to_remove_from_this_entity = sample_to_explain_2_best_rule[cur_sample_to_explain]
            for x in range(4):
                if x < len(samples_to_remove_from_this_entity):
                    b.append(";".join(dataset.sample_to_fact(samples_to_remove_from_this_entity[x])))
                else:
                    b.append(";;")

            b = ";".join(b)
            c = str(original_direct_score) + ";" + str(new_direct_score)
            d = str(original_tail_rank) + ";" + str(new_tail_rank)
            output_lines.append(";".join([a, b, c, d]) + "\n")

            statistics["samples"] += 1
            statistics["original_direct_score"] += original_direct_score
            statistics["new_direct_score"] += new_direct_score
            statistics["original_tail_rank"] += original_tail_rank
            statistics["new_tail_rank"] += new_tail_rank
            statistics["score_change_list"] += [original_direct_score - new_direct_score]
            statistics["rank_change_list"] += [new_tail_rank - original_tail_rank]


        print(log_end_to_end_output_path)
        with open(log_end_to_end_output_path, "w", encoding="utf8" ) as outfile:
            outfile.writelines(output_lines)
    else:
        raise ValueError(f"Mode '{mode}' not recognized. \n"
                         "Try one of: {'sufficient', 'necessary'}")   
    try:
        path_statistics = os.path.join(STATISTICS_PATH, f"{dataset.name}_statistics.csv")
        print(path_statistics)
        statistics["original_direct_score"] = statistics["original_direct_score"] / statistics["samples"]
        statistics["new_direct_score"] = statistics["new_direct_score"] / statistics["samples"]
        statistics["original_tail_rank"] = statistics["original_tail_rank"] / statistics["samples"]
        statistics["new_tail_rank"] = statistics["new_tail_rank"] / statistics["samples"]

        statistics["rank_change_mean"] = sum(statistics["rank_change_list"]) / statistics["samples"]
        statistics["score_change_mean"] = sum(statistics["score_change_list"]) / statistics["samples"]

        statistics["rank_change_median"] = numpy.median(statistics["rank_change_list"])
        statistics["score_change_median"] = numpy.median(statistics["score_change_list"])

        print(statistics)
        with open(log_path, "a", encoding="utf8" ) as execution_log:
            execution_log.write(str(statistics))
        
        df = pd.read_csv(path_statistics, sep=";", decimal=".")
        if 'score_change_median' not in df:
            for col in list(statistics.keys()):
                df[col] = ""
        row = list(df.iloc[-1,:-len(statistics.keys())])
        row.extend(list(statistics.values()))
        new_df = df.iloc[:-1,:]
        new_df.loc[len(new_df)] = row
        new_df.to_csv(path_statistics, sep=';', index=False)

        # delete explanations to verify file to avoid reuse in other experiments
        if os.path.exists(explanations_path):
            os.remove(explanations_path)
        
    except Exception as e:
        print(statistics)
        print(e)
