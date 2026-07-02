import collections
import os
import sys

sys.path.append(os.path.realpath(
    os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))

from src.model.dataset import Dataset
from src.relevance_engines.post_training_engine import PostTrainingEngine
from src.link_prediction.models.model import Model
from src.explanation_builders.explanation_builder import SufficientExplanationBuilder
from src.model.ruleset import RuleEvidence, Ruleset
from src.model.utils import flatten_list

MODE = "sufficient"


class SymbolicSufficientExplanationBuilder(SufficientExplanationBuilder):
    """
    The SymbolicSufficientExplanationBuilder object guides the search for sufficient triples based on rules
    """

    def __init__(self, model: Model,
                 dataset: Dataset,
                 ruleset: Ruleset,
                 hyperparameters: dict,
                 sample_to_explain: tuple[int, int, int],
                 perspective: str,
                 output_prefix: str,
                 entities_to_convert: list,
                 num_entities_to_convert: int = 10,
                 relevance_threshold: float = None,
                 max_explanation_length: int = -1,
                 verbose: bool = True):
        """
        SymbolicSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain: the predicted sample to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param num_entities_to_convert
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(model=model, dataset=dataset,
                         sample_to_explain=sample_to_explain, perspective=perspective,
                         num_entities_to_convert=num_entities_to_convert,
                         max_explanation_length=max_explanation_length,
                         output_prefix=output_prefix)

        engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)
        
        # compute complexity -> number of rules checked for relevance
        self.verbose = verbose
        self.length_cap = 5 #TODO: what value best and make arg?

        if entities_to_convert is not None:
            self.entities_to_convert = entities_to_convert
        else:
            self.entities_to_convert = engine.extract_entities_for(model=self.model,
                                                                        dataset=self.dataset,
                                                                        sample=sample_to_explain,
                                                                        perspective=perspective,
                                                                        k=num_entities_to_convert,
                                                                        degree_cap=200)
            
        
        self.ruleEvidence = RuleEvidence(ruleset=ruleset, dataset=dataset, engine=engine, perspective=perspective, mode=MODE,
        sample_to_explain=sample_to_explain, output_prefix=output_prefix, relevance_threshold=relevance_threshold, entities_to_convert=self.entities_to_convert, verbose=verbose, perspective_entity=self.perspective_entity)

    def build_explanations(self,
                           rules_to_add: list[list[list[int, int, int]]],
                           top_k: int = 10, algorithm="rules_kelpie"):
        
        self.ruleEvidence.set_rules_to_remove(rules_to_add, top_k)

        # identify Algorithm to run
        algorithm_map = {
            "reverse": self.kelpie_rule_reverse_alg,
            "pca": self.pca_rule_alg,
            "frequency": self.frequency_rule_alg,
            "kelpie": self.kelpie_rule_alg
        }
        
        matching_keys = [key for key in algorithm_map if key in algorithm.lower()]
        if len(matching_keys) > 1:
            raise ValueError(f"Multiple matching keys: {matching_keys}\n"
                             f"Found in algorithm '{algorithm}'.")
        alg_func = algorithm_map[matching_keys[0]] if matching_keys else None
        
        if alg_func is None:
            raise ValueError(f"Algorithm '{algorithm}' not recognized. \n"
                             f"Try one of: {', '.join(algorithm_map.keys())}")
        
        # execute algorithm identified above
        all_rules_with_relevance, complexity = alg_func(rules_to_add)
        
        return sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True), complexity
    

    def kelpie_rule_alg(self, rules_to_remove):
        print(f"\n----- Triple Relevance: {len(self.ruleEvidence.relevant_triples_set)} unique triples -------")
        # get relevance for each triple, for later calculation of preliminary score
        sample_2_relevance = self.ruleEvidence.extract_sample_relevance(self.ruleEvidence.relevant_triples_set)
        
        print(f"\n----- Single Rule Relevance: {len(rules_to_remove)} different rules -------")
        # get relevance for all rules
        rule_2_preliminary_relevance = self.ruleEvidence.extract_rule_relevance()
        print(rule_2_preliminary_relevance)
        
        self.ruleEvidence.short_print_rule_relevance(rule_2_preliminary_relevance)

        # combined rule relevance (print in method)
        all_rules_with_relevance, complexity = self.ruleEvidence.extract_rule_relevance_combinatorial(self.length_cap)

        return all_rules_with_relevance, complexity
    

    def kelpie_rule_reverse_alg(self, rules_to_remove: list[list[list[int]]]):
        print(f"\n----- Full Set Relevance: {len(rules_to_remove)} rules -------")
        # get relevance for the full rule set
        full_set_relevance = self.ruleEvidence.extract_set_relevance(rules_to_remove)
        
        # if only one rule applicable, onlyone relevance calculation is sufficient
        if len(rules_to_remove) == 1:
            return [[rules_to_remove, full_set_relevance]], self.ruleEvidence.get_complexity()

        print(f"\n----- Partial Set Relevance: {len(rules_to_remove)} versions -------")
        # get relevance for ruleset 
        sufficient_set = self.ruleEvidence.extract_partial_set_relevance(full_set_relevance)

        print(f"\n----- Sufficient Set Relevance: {len(sufficient_set)} rules -------")
        # get relevance for the full rule set
        nec_set_relevance = self.ruleEvidence.extract_set_relevance(sufficient_set)

        return [(tuple(sufficient_set), nec_set_relevance)], self.ruleEvidence.get_complexity()
    

    def pca_rule_alg(self, rules_to_remove):
        print(f"\n----- Single Rule PCA Confidence: {len(rules_to_remove)} different rules -------")
        # get relevance for all rules
        rule_2_confidence = self.ruleEvidence.extract_rule_confidence(rules=rules_to_remove)

        # sort by relevance and store as list
        rule_confidence_list = sorted(rule_2_confidence.items(), key=lambda x: x[1], reverse=True)
        self.ruleEvidence.short_print_rule_relevance(rule_confidence_list)

        print(f"\n----- Combined Rule Relevance: max length {len(rules_to_remove)} Rules -------")
        # get relevance incrementally
        all_rules_with_relevance, complexity = \
            self.ruleEvidence.extract_rule_relevance_incrementally(rule_confidence_list)
        #self.ruleEvidence.short_print_rule_relevance(all_rules_with_relevance)

        return all_rules_with_relevance, complexity


    def frequency_rule_alg(self, rules_to_remove):
        triples = flatten_list(rules_to_remove)
        print(f"\n----- Single triple frequency: {len(set(triples))} unique triples in rules -------")
        # get relevance for all rules
        triple_2_frequency = collections.Counter(triples)

        # sort by relevance and store as list
        triple_frequency_list = sorted(triple_2_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # pack triples in list, so later handling works
        for i in range(len(triple_frequency_list)):
            triple, value = triple_frequency_list[i]
            triple_frequency_list[i] = ([triple], value)

        self.ruleEvidence.short_print_rule_relevance(triple_frequency_list)

        print(f"\n----- Combined Triple Relevance: max length {len(triple_frequency_list)} Triples -------")
        # get relevance incrementally
        all_rules_with_relevance, complexity = \
            self.ruleEvidence.extract_rule_relevance_incrementally(triple_frequency_list)
        #self.ruleEvidence.short_print_rule_relevance(all_rules_with_relevance)

        return all_rules_with_relevance, complexity
    