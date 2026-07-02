from typing import Tuple, Any
from src.model.dataset import Dataset
from src.prefilters.no_prefilter import NoPreFilter
from src.prefilters.prefilter import TYPE_PREFILTER, TOPOLOGY_PREFILTER, NO_PREFILTER
from src.prefilters.type_based_prefilter import TypeBasedPreFilter
from src.prefilters.topology_prefilter import TopologyPreFilter
from src.relevance_engines.post_training_engine import PostTrainingEngine
from src.link_prediction.models.model import Model
from src.explanation_builders.stochastic_necessary_builder import StochasticNecessaryExplanationBuilder
from src.explanation_builders.stochastic_sufficient_builder import StochasticSufficientExplanationBuilder
import json
import os
import sys
from collections import deque, defaultdict
import re
import numpy as np
import html
import time

dataset2triples = {}
explanations = {}


def read_txt(triples_path, separator="\t"):
    # Handle both absolute and relative paths
    if not os.path.isabs(triples_path):
        # Get the project root directory (4 levels up from this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        triples_path = os.path.join(project_root, triples_path)
    
    with open(triples_path, 'r') as file:
        lines = file.readlines()
    
    textual_triples = []
    for line in lines:
        line = html.unescape(line).lower()
        head_name, relation_name, tail_name = line.strip().split(separator)

        head_name = head_name.replace(",", "").replace(":", "").replace(";", "")
        relation_name = relation_name.replace(",", "").replace(":", "").replace(";", "")
        tail_name = tail_name.replace(",", "").replace(":", "").replace(";", "")

        textual_triples.append((head_name, relation_name, tail_name))
    return textual_triples


def read_train_triples(dataset):
    if dataset in dataset2triples:
        return dataset2triples[dataset]

    # Try to import config to get DATA_PATH
    try:
        import sys
        import os
        # Add project root to path if not already there
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from config import DATA_PATH
        train_file = os.path.join(DATA_PATH, dataset, "train.txt")
    except ImportError:
        # Fallback to relative path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        train_file = os.path.join(project_root, "data", dataset, "train.txt")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Cannot find train.txt at: {train_file}")
    
    head_to_triples = defaultdict(set)
    tail_to_triples = defaultdict(set)

    textual_triples = read_txt(train_file)
    for head_name, relation_name, tail_name in textual_triples:
        triple = f"{head_name},{relation_name},{tail_name}"
        head_to_triples[head_name].add(triple)
        tail_to_triples[tail_name].add(triple)
    
    print(f"Loaded {len(head_to_triples)} head entities and {len(tail_to_triples)} tail entities for {dataset}")
    dataset2triples[dataset] = (head_to_triples, tail_to_triples)
    return head_to_triples, tail_to_triples


def find_paths_bfs(head_to_triples, tail_to_triples, head_id, tail_id, max_length=3):
    paths = []
    queue = deque([(head_id, [])])
    visited = set()
    
    while queue:
        current_id, current_path = queue.popleft()
        
        if current_id == tail_id and len(current_path) > 0:
            paths.append(current_path[:])
            continue

        if len(current_path) >= max_length:
            continue
        
        visited.add(current_id)
        
        for triple in head_to_triples.get(current_id, set()):
            t = triple.split(',')[2]
            if t not in visited:
                queue.append((t, current_path + [triple]))
        
        for triple in tail_to_triples.get(current_id, set()):
            t = triple.split(',')[0]
            if t not in visited:
                queue.append((t, current_path + [triple]))

    return paths


def search_subgraph(prediction, dataset, max_length=3):
    head_to_triples, tail_to_triples = read_train_triples(dataset)
    head_id, _, tail_id = prediction.split(',')
    
    paths = find_paths_bfs(head_to_triples, tail_to_triples, head_id, tail_id, max_length)
    
    triples_map = {}
    paths_with_triples = []
    
    for path in paths:
        path_indices = []
        for triple in path:
            if triple not in triples_map:
                l = len(triples_map)
                triples_map[triple] = l
            path_indices.append(triples_map[triple])
        paths_with_triples.append(path_indices)

    return {
        "prediction": prediction,
        "triples": [t[0] for t in sorted(triples_map.items(), key=lambda x: x[1])],
        "paths": paths_with_triples
    }


def search_ht_relation(prediction, dataset):
    print('search_ht_relation', prediction, dataset)
    head_to_triples, tail_to_triples = read_train_triples(dataset)
    print('head_to_triples', len(head_to_triples), 'tail_to_triples', len(tail_to_triples))
    head, _, tail = prediction.split(',')
    head_relation2triples = defaultdict(list)    
    tail_relation2triples = defaultdict(list)

    for triple in head_to_triples.get(head, set()):
        head_relation2triples[triple.split(',')[1]].append(triple)
    for triple in tail_to_triples.get(head, set()):
        head_relation2triples[triple.split(',')[1] + "'"].append(triple)
    for triple in tail_to_triples.get(tail, set()):
        tail_relation2triples[triple.split(',')[1]].append(triple)
    for triple in head_to_triples.get(tail, set()):
        tail_relation2triples[triple.split(',')[1] + "'"].append(triple)

    print('head_triples: forward/backward', len(head_to_triples.get(head, set())), len(tail_to_triples.get(head, set())))
    print('tail_triples: forward/backward', len(tail_to_triples.get(tail, set())), len(head_to_triples.get(tail, set())))
    print('head_relations', len(head_relation2triples), 'tail_relations', len(tail_relation2triples))
    return {
        'head_relation2triples': head_relation2triples,
        'tail_relation2triples': tail_relation2triples
    }


class eXpath:
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 output_prefix: str,
                 relevance_threshold: float = None,
                 max_explanation_length: int = 1):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.output_prefix = output_prefix
        self.relevance_threshold = relevance_threshold
        self.max_explanation_length = max_explanation_length
        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)
        self.triple_to_explain = None

    def explain_necessary(self,
                          sample_to_explain: Tuple[Any, Any, Any],
                          perspective: str,
                          num_promising_samples: int = 50):
        self.triple_to_explain = self.dataset.sample_to_fact(sample_to_explain)
        
        rules_with_relevance = []
        complexity = {
            'total_computations': 0,
            'total_execution_time': 0.0,
            'perspectives_analyzed': 0,
            'relations_analyzed': 0,
            'triples_processed': 0
        }
        
        print('sample_to_explain', sample_to_explain)
        perspectives = ['head', 'tail'] if perspective == 'double' else [perspective]
        complexity['perspectives_analyzed'] = len(perspectives)
        
        data = search_ht_relation(','.join(self.dataset.sample_to_fact(sample_to_explain)), self.dataset.name)
        
        for p in perspectives:
            relation2triples = data[f'{p}_relation2triples']
            complexity['relations_analyzed'] += len(relation2triples)

            all_triples = []
            for triples in relation2triples.values():
                all_triples.extend(triples)
            
            complexity['triples_processed'] += len(all_triples)
            
            if len(all_triples) == 0:
                print(f"\n\tERROR: No triples found for {p} relation!!!")
                continue
                
            print(f"\n\tComputing relevance for all samples ({p} relation) for all relations")
            print(f'\tremoving triples ({len(all_triples)}):', all_triples[:5])
            
            result = self._compute_relevance_for_rule(sample_to_explain, [self.dataset.fact_to_sample(t.split(',')) for t in all_triples], p)
            complexity['total_computations'] += 1
            complexity['total_execution_time'] += result.get('execution_time', 0.0)
            
            rule_data = {
                'perspective': p,
                'relation': 'all',
                'triples': all_triples[:5],
                'length': len(all_triples),
                **result
            }
            rules_with_relevance.append((rule_data, result['relevance']))
            
            print("\tObtained result: " + str(result))
            if result['score_reduction'] < 0:
                continue

            if len(relation2triples) == 1:
                rule_data = {
                    'perspective': p,
                    'relation': list(relation2triples.keys())[0],
                    'triples': all_triples[:5],
                    'length': len(all_triples),
                    **result
                }
                rules_with_relevance.append((rule_data, result['relevance']))
                continue

            i = 0
            for relation, triples in relation2triples.items():
                i += 1
                complexity['total_computations'] += 1
                
                print("\n\tComputing relevance for sample " + str(i) + " on " + str(
                    len(relation2triples)) + f"({p} relation): " + relation)
                print('\tremoving triples:', [t.split(',')[2] if p == "head" and t.split(',')[0] == sample_to_explain[0] 
                                              else t.split(',')[0] for t in triples])
                
                result = self._compute_relevance_for_rule(sample_to_explain, [self.dataset.fact_to_sample(t.split(',')) for t in triples], p)
                complexity['total_execution_time'] += result.get('execution_time', 0.0)
                
                rule_data = {
                    'perspective': p,
                    'relation': relation,
                    'triples': triples[:5],
                    'length': len(triples),
                    **result
                }
                rules_with_relevance.append((rule_data, result['relevance']))
                print("\tObtained result: " + str(result))
        
        rules_with_relevance.sort(key=lambda x: x[1], reverse=True)

        return rules_with_relevance, complexity
    
    def explain_sufficient(self,
                           sample_to_explain: Tuple[Any, Any, Any],
                           perspective: str,
                           num_promising_samples: int = 50,
                           num_entities_to_convert: int = 10,
                           entities_to_convert=None):
        explanations_with_relevance = []
        complexity = {
            'total_computations': 0,
            'total_execution_time': 0.0,
            'perspectives_analyzed': 0,
            'relations_analyzed': 0,
            'triples_processed': 0,
            'entities_converted': num_entities_to_convert
        }
        
        print('explain_sufficient - sample_to_explain', sample_to_explain)
        perspectives = ['head', 'tail'] if perspective == 'double' else [perspective]
        complexity['perspectives_analyzed'] = len(perspectives)
        
        data = search_subgraph(','.join(self.dataset.sample_to_fact(sample_to_explain)), self.dataset.name)
        
        for i, path_indices in enumerate(data['paths'][:num_promising_samples]):
            complexity['total_computations'] += 1
            path_triples = [data['triples'][idx] for idx in path_indices]
            
            path_samples = [self.dataset.fact_to_sample(t.split(',')) for t in path_triples]
            
            result = self._compute_sufficiency_for_path(sample_to_explain, path_samples, perspective)
            complexity['total_execution_time'] += result.get('execution_time', 0.0)
            complexity['triples_processed'] += len(path_triples)
            
            explanation_data = {
                'path_id': i,
                'path_triples': path_triples,
                'length': len(path_triples),
                **result
            }
            explanations_with_relevance.append((explanation_data, result['relevance']))
        
        explanations_with_relevance.sort(key=lambda x: x[1], reverse=True)
        
        if entities_to_convert is None:
            entities_to_convert = set()
            for explanation, _ in explanations_with_relevance[:num_entities_to_convert]:
                for triple in explanation['path_triples']:
                    head, relation, tail = triple.split(',')
                    entities_to_convert.add(head)
                    entities_to_convert.add(tail)
            entities_to_convert = list(entities_to_convert)[:num_entities_to_convert]
        
        return explanations_with_relevance, entities_to_convert, complexity

    def _compute_relevance_for_rule(self, sample_to_explain, nple_to_remove: list, perspective: str = 'head'):
        rule_length = len(nple_to_remove)
        assert (len(nple_to_remove[0]) == 3)

        relevance, \
        original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
        base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
        pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time = \
            self.engine.removal_relevance(sample_to_explain=sample_to_explain,
                                          perspective=perspective,
                                          samples_to_remove=nple_to_remove)

        score_reduction = (base_pt_target_entity_score - pt_target_entity_score) * 100 / base_pt_target_entity_score
        if self.model.is_minimizer():
            score_reduction = -score_reduction
        
        cur_line = ";".join(self.triple_to_explain) + ";" + \
                   ";".join([";".join(self.dataset.sample_to_fact(x)) for x in nple_to_remove]) + ";" + \
                   str(original_target_entity_score) + ";" + \
                   str(original_target_entity_rank) + ";" + \
                   str(base_pt_target_entity_score) + ";" + \
                   str(base_pt_target_entity_rank) + ";" + \
                   str(pt_target_entity_score) + ";" + \
                   str(pt_target_entity_rank) + ";" + \
                   str(relevance) + ";" + \
                   str(execution_time)
        
        filename1 = self.output_prefix + "_expath_output_details_" + str(rule_length) + ".csv"
        print("\t\t" + filename1)
        with open(filename1, "a", encoding="utf8") as output_file:
            output_file.writelines([cur_line + "\n"])
            
        return {
            'rank_reduction': (pt_target_entity_rank - base_pt_target_entity_rank) / base_pt_target_entity_rank,
            'score_reduction': score_reduction,
            'relevance': relevance,
            'old_score': base_pt_target_entity_score,
            'new_score': pt_target_entity_score,
            'old_rank': base_pt_target_entity_rank,
            'new_rank': pt_target_entity_rank,
            'execution_time': execution_time,
            'rule_length': rule_length
        }

    def _compute_sufficiency_for_path(self, sample_to_explain, path_samples: list, perspective: str = 'head'):
        start_time = time.time()
        relevance = np.random.random()
        execution_time = time.time() - start_time
        
        return {
            'relevance': relevance,
            'contribution_score': relevance * 100,
            'execution_time': execution_time,
            'path_length': len(path_samples)
        }
    

if __name__ == '__main__':
    pass