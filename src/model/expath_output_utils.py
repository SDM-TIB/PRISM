"""
FILE 2 of 4: expath_output_utils.py
Install to: src/model/expath_output_utils.py (NEW FILE)

Helper functions for formatting and saving eXpath explanation results.
Copy this entire file and save it as src/model/expath_output_utils.py
"""

import json
from typing import List, Tuple, Any


def save_expath_explanation_necessary(sample_to_explain: Tuple[Any, Any, Any],
                                       rules_with_relevance: List[Tuple[Any, float]],
                                       dataset,
                                       output_file: str):
    """
    Save necessary explanation in the format expected by verify_explanations.py
    
    Format:
    Line 1: head;relation;tail (sample to explain)
    Line 2: triple1;triple2;triple3:relevance|triple4;triple5:relevance|...
    Line 3: (empty)
    
    Handles both formats:
    - eXpath: (rule_data_dict, relevance)
    - Criage/Kelpie: (rule_samples_list, relevance)
    """
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # Line 1: Sample to explain
        if isinstance(sample_to_explain[0], int):
            fact = dataset.sample_to_fact(sample_to_explain)
        else:
            fact = sample_to_explain
        
        f.write(";".join(fact) + "\n")
        
        # Line 2: Rules with relevance
        formatted_rules = []
        
        for rule_data, relevance in rules_with_relevance:
            # Handle both dict format (eXpath) and list format (Criage/Kelpie)
            if isinstance(rule_data, dict):
                # eXpath format: rule_data is a dictionary with 'triples' key
                triples = rule_data.get('triples', [])
                
                if not triples:
                    continue
                
                # Triples are already in "head,rel,tail" format
                rule_str = ";".join(triples)
            elif isinstance(rule_data, (list, tuple)):
                # Criage/Kelpie format: rule_data is a list of samples
                if not rule_data:
                    continue
                
                # Convert samples to "head,rel,tail" format
                triple_strs = []
                for sample in rule_data:
                    fact = dataset.sample_to_fact(sample)
                    triple_strs.append(",".join(fact))
                
                rule_str = ";".join(triple_strs)
            else:
                # Unknown format, skip
                print(f"Warning: Unknown rule_data format: {type(rule_data)}")
                continue
            
            formatted_rules.append(f"{rule_str}:{relevance}")
        
        if formatted_rules:
            f.write("|".join(formatted_rules) + "\n")
        else:
            f.write("\n")
        
        # Line 3: Empty line
        f.write("\n")


def save_expath_explanation_sufficient(sample_to_explain: Tuple[Any, Any, Any],
                                        explanations_with_relevance: List[Tuple[Any, float]],
                                        entities_to_convert: List[Any],
                                        dataset,
                                        output_file: str):
    """
    Save sufficient explanation in the format expected by verify_explanations.py
    
    Format:
    Line 1: head;relation;tail (sample to explain)
    Line 2: entity1,entity2,entity3,... (entities to convert)
    Line 3: triple1;triple2;triple3:relevance|triple4;triple5:relevance|...
    Line 4: (empty)
    
    Handles both formats:
    - eXpath: (explanation_data_dict, relevance)
    - Criage/Kelpie: (rule_samples_list, relevance)
    """
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # Line 1: Sample to explain
        if isinstance(sample_to_explain[0], int):
            fact = dataset.sample_to_fact(sample_to_explain)
        else:
            fact = sample_to_explain
        
        f.write(";".join(fact) + "\n")
        
        # Line 2: Entities to convert
        entity_names = []
        for entity in entities_to_convert:
            if isinstance(entity, int):
                entity_names.append(dataset.entity_id_2_name[entity])
            else:
                entity_names.append(entity)
        f.write(",".join(entity_names) + "\n")
        
        # Line 3: Explanations with relevance
        formatted_explanations = []
        
        for explanation_data, relevance in explanations_with_relevance:
            # Handle both dict format (eXpath) and list format (Criage/Kelpie)
            if isinstance(explanation_data, dict):
                # eXpath format: explanation_data is a dictionary
                path_triples = explanation_data.get('path_triples', [])
                
                if not path_triples:
                    continue
                
                explanation_str = ";".join(path_triples)
            elif isinstance(explanation_data, (list, tuple)):
                # Criage/Kelpie format: explanation_data is a list of samples
                if not explanation_data:
                    continue
                
                # Convert samples to "head,rel,tail" format
                triple_strs = []
                for sample in explanation_data:
                    fact = dataset.sample_to_fact(sample)
                    triple_strs.append(",".join(fact))
                
                explanation_str = ";".join(triple_strs)
            else:
                print(f"Warning: Unknown explanation_data format: {type(explanation_data)}")
                continue
            
            formatted_explanations.append(f"{explanation_str}:{relevance}")
        
        if formatted_explanations:
            f.write("|".join(formatted_explanations) + "\n")
        else:
            f.write("\n")
        
        # Line 4: Empty line
        f.write("\n")


def save_expath_results_json(sample_to_explain: Tuple[Any, Any, Any],
                              rules_with_relevance: List[Tuple[dict, float]],
                              complexity: dict,
                              dataset,
                              output_file: str):
    """
    Save eXpath results as JSON for easier debugging and analysis.
    """
    
    if isinstance(sample_to_explain[0], int):
        fact = dataset.sample_to_fact(sample_to_explain)
    else:
        fact = sample_to_explain
    
    data = {
        'sample_to_explain': {
            'head': fact[0],
            'relation': fact[1],
            'tail': fact[2]
        },
        'rules': [],
        'complexity': complexity
    }
    
    for rule_data, relevance in rules_with_relevance:
        data['rules'].append({
            'triples': rule_data.get('triples', []),
            'relevance': float(relevance),
            'perspective': rule_data.get('perspective', ''),
            'relation': rule_data.get('relation', ''),
            'length': rule_data.get('length', 0),
            'score_reduction': float(rule_data.get('score_reduction', 0)),
            'rank_reduction': float(rule_data.get('rank_reduction', 0)),
            'old_rank': int(rule_data.get('old_rank', 0)),
            'new_rank': int(rule_data.get('new_rank', 0)),
            'old_score': float(rule_data.get('old_score', 0)),
            'new_score': float(rule_data.get('new_score', 0)),
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_expath_results_json(input_file: str):
    """
    Load eXpath results from JSON format.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_to_explain = (
        data['sample_to_explain']['head'],
        data['sample_to_explain']['relation'],
        data['sample_to_explain']['tail']
    )
    
    rules_with_relevance = []
    for rule in data['rules']:
        rule_data = {
            'triples': rule['triples'],
            'perspective': rule['perspective'],
            'relation': rule['relation'],
            'length': rule['length'],
            'score_reduction': rule['score_reduction'],
            'rank_reduction': rule['rank_reduction'],
            'old_rank': rule['old_rank'],
            'new_rank': rule['new_rank'],
            'old_score': rule['old_score'],
            'new_score': rule['new_score'],
        }
        rules_with_relevance.append((rule_data, rule['relevance']))
    
    complexity = data.get('complexity', {})
    
    return sample_to_explain, rules_with_relevance, complexity