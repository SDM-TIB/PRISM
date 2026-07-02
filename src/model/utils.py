from inspect import getargvalues, stack
from pathlib import Path
import copy
import os
import sys
import time
import pandas as pd

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from config import LOGGING_PATH, RANKS_PATH, PREDICTIONS_PATH
from src.model.dataset import Dataset

def extract_good_predictions(source_file:str,  dataset:str=None, output_file:str=None, embedding:str="TransE", seeds:list[int]=[42], min_rank:int=3, mode:str="RND", n:int=100, replace:bool=False, direction:str="tail"):
    """
    extracts n lines from the raw file of "filtered_ranks.csv". 
    Mode RND extracts them randomly, depending on the seed and writes as many files as seeds are given. 
    Mode TOP returns the top n lines, sorted by tail/head rank.
    if there are less than n predictions with rank 1, then the outputfile is smaller. reducing the min_rank to 3 for example, gives access to more "good" predictions to use.

    Args:
        source_file (str): file with table of predictions and rank, usually "{dataset}/filtered_ranks.csv"
        output_file (str): filename to write "predictions" to. eg. data/4_predictions/transe_{dataset}.csv
        seeds (list, optional): list of seeds for random mode, as many input_fact files as seeds. Defaults to [42].
        min_rank (int, optional): minimum rank of predictions which are considered for selection. rank 1 are correct predictions, if there are less than n of these, rank should be reduced. Defaults to 3.
        mode (str, optional): random selection or top n lines. Defaults to "RND".
        replace (bool, optional): false is recommended, this prevents duplicate lines in the output file. Defaults to False.
        direction (str, optional): "head" or "tail" prediction. Defaults to "tail".

    returns:
        last file name of created predictions
    """ 
    args = arguments()

    if output_file == None:
        output_file = f"{embedding}_{dataset}".lower()

    if type(seeds) == int:
        seeds = [seeds]
    
    if (len(seeds)>1) & (mode=="TOP"):
        print("INFO: caution, giving several seeds doesn't change the output of top_n rows. only one file will be created")

    input_path = os.path.join(RANKS_PATH, source_file)
    predictions_path = os.path.join(PREDICTIONS_PATH, output_file)
    output_prefix, timestamp = generate_output_file_prefix(embedding=embedding)

    df = pd.read_csv(input_path, sep=';', header=0, names=["s", "p", "o", "head_rank", "tail_rank"])
    
    df_filtered = df[df[f"{direction}_rank"] <= min_rank]
    
    for seed in seeds:
        if mode == "RND":
            df_subset = df_filtered.sample(n=n, replace=replace, random_state=seed) #randomly select n rows. with or without repeat
            df_subset = df_subset.iloc[:, :3]
        elif mode == "TOP":
            df_subset = df_subset.sort_values(by=f"{direction}_rank", ascending=[True])
            df_subset = df_filtered.iloc[:n, :3]

        # output files
        # data
        filename = f"{predictions_path}_{direction}_{seed}.csv"
        df_subset.to_csv(filename, sep='\t', index=False, header=False)
        
        # data logging
        filename_7_output = f"{output_prefix}_7_filtered_ranks_{direction}_{seed}.csv"
        df_subset.to_csv(filename_7_output, sep='\t', index=False, header=False)

    #logging
    filename_6_output = f"{output_prefix}__6_predictions_logging.txt"
    args["output_path"] = filename
    with open(filename_6_output, "w", encoding="utf8" ) as execution_log:
        execution_log.write(f"test.py at {timestamp}\n")
        execution_log.write(str(args)+"\n")

    print(f"{direction} length: {len(df_filtered)}, sample: {len(df_subset)}")
    print(filename_7_output)
    return filename

@staticmethod
def arguments():
    """Returns dictionary of calling function's
        named arguments, that existed at that time.
    """
    args = getargvalues(stack()[1][0])[-1:]
    return copy.deepcopy(args[0])

@staticmethod
def generate_output_file_prefix(embedding, mode="necessary"):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    day_folder = time.strftime("%Y-%m-%d")
    path = os.path.join(LOGGING_PATH, day_folder)
    Path(path).mkdir(parents=True, exist_ok=True)
    output_prefix = os.path.join(path, timestamp + f"_{embedding}_{mode[0]}")
    return output_prefix, timestamp

@staticmethod
def concat_output(rule_samples_with_relevance, dataset, fact, verbose, entities_to_convert:list = None)-> str:
    output_list = []
    rule_facts_with_relevance = []
    for cur_rule_with_relevance in rule_samples_with_relevance:
        cur_rule_samples, cur_relevance = cur_rule_with_relevance

        cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
        cur_rule_facts = ";".join([",".join(x) for x in cur_rule_facts])
        rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

    if verbose:
        print(";".join(fact))
        print(",\n ".join(rule_facts_with_relevance))
        print()
    output_list.append(";".join(fact) + "\n")
    if entities_to_convert != None:
        output_list.append(", ".join(entities_to_convert) + "\n")
    output_list.append("|".join(rule_facts_with_relevance) + "\n")
    output_list.append("\n")
    output_lines="".join(output_list)

    return output_lines

    
@staticmethod
def flatten_list(list_of_lists:list[list[tuple[int, int, int]]])-> list[tuple[int, int, int]]:
    flat_list = []
    for li in list_of_lists:
        for triple in li:
            flat_list.append(triple)
    return flat_list

@staticmethod
#def seperate_rule_triples(long_rules):
    #triple_rule = []
    #for rule in long_rules:
        #assert len(rule) % 3 == 0, f"rule is not made of triples, please check output.csv for rule: {rule}"
        #i = 0
        #while i < len(rule):
            #triple_rule.append([rule[i], rule[i+1], rule[+2]])
            #i += 3
    
    #return triple_rule


def seperate_rule_triples(long_rules):
    triple_rule = []
    for rule in long_rules:
        # Check if rule is a list of single characters (indicating a string was split incorrectly)
        if (isinstance(rule, list) and len(rule) > 3 and 
            all(isinstance(x, str) and len(x) == 1 for x in rule)):
            # This looks like a string that was split into characters - rejoin it
            rejoined_rule = ''.join(rule)
            print(f"Warning: Found rule split into characters: {rule}")
            print(f"Rejoining as: '{rejoined_rule}'")
            # Treat the rejoined string as a single element (not a triple)
            triple_rule.append([rejoined_rule])
            continue
        
        # Check if rule is a single string (should be treated as one element)
        if isinstance(rule, str):
            triple_rule.append([rule])
            continue
            
        assert len(rule) % 3 == 0, f"rule is not made of triples, please check output.csv for rule: {rule}"
        i = 0
        while i < len(rule):
            triple_rule.append([rule[i], rule[i+1], rule[i+2]])  # Fixed: changed rule[+2] to rule[i+2]
            i += 3
    
    return triple_rule

# never used, remove?
@staticmethod
def extract_relations(rules:list[list[tuple[int, int, int]]]|list[tuple[int, int, int]], dataset:Dataset) -> list[str]|list[list[str]]:
    relation_list = []
    for i, group in enumerate(rules):
        if (len(group) == 3) & (type(group[0]) == int):     # group:tuple[int, int, int]
            sample = dataset.sample_to_fact(group)          # sample:tuple[str, str, str]
            relation_list.append(sample[1])                 # only append predicate
        elif (len(group[0]) == 3) & (type(group[0][0]) == int):
            relation_list.append([])
            for tuple in group:               # tuple:tuple[int, int, int]
                sample = dataset.sample_to_fact(tuple)      # sample:tuple[str, str, str]
                relation_list[i].append(sample[1])          # only append predicate
        else:
            relation_list.append(f"malformed group '{group}'")
    return relation_list


if __name__ == "__main__":
    dataset = "FR_Reduced_2K"
    filtered_ranks = f"{dataset}/TransE_filtered_ranks.csv"
    write_filename = f"TransE_{dataset}".lower()

    extract_good_predictions(source_file=filtered_ranks, output_file=write_filename, min_rank=1, mode="RND", seeds=[42])