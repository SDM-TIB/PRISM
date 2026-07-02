from typing import Tuple, Any
from src.model.dataset import Dataset
from src.relevance_engines.criage_engine import CriageEngine
from src.link_prediction.models.model import Model
from src.explanation_builders.explanation_builder import NecessaryExplanationBuilder

class CriageNecessaryExplanationBuilder(NecessaryExplanationBuilder):

    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any],
                 perspective: str,
                 output_prefix: str):
        
        super().__init__(model, dataset, sample_to_explain, perspective, 1, output_prefix)

        self.engine = CriageEngine(model=model,
                                   dataset=dataset,
                                   hyperparameters=hyperparameters)
        
        # Initialize complexity tracking like in stochastic builder
        self.complexity: dict = {}

    def build_explanations(self,
                           samples_to_remove: list,
                           top_k: int = 10):

        rule_2_relevance = {}
        (head_to_explain, _, tail_to_explain) = self.sample_to_explain

        for i, sample_to_remove in enumerate(samples_to_remove):
            print("\n\tComputing relevance for sample " + str(i) + " on " + str(len(samples_to_remove)) + ": " +
                  self.dataset.printable_sample(sample_to_remove))

            # Track complexity - each sample is a rule of length 1
            rule_length = 1  # Since we're processing individual samples
            if rule_length in self.complexity:
                self.complexity[rule_length] += 1
            else:
                self.complexity[rule_length] = 1

            tail_to_remove = sample_to_remove[2]

            if tail_to_remove == head_to_explain:
                perspective = "head"
            elif tail_to_remove == tail_to_explain:
                perspective = "tail"
            else:
                raise ValueError

            relevance = self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                                    perspective=perspective,
                                                    samples_to_remove=[sample_to_remove])

            rule_2_relevance[tuple([sample_to_remove])] = relevance

            cur_line = ";".join(self.triple_to_explain) + ";" + \
                        ";".join(self.dataset.sample_to_fact(sample_to_remove)) + ";" \
                       + str(relevance)
            
            # Fix the string concatenation
            filename1 = self.output_prefix + "_output_details_1.csv"
            print(filename1)
            with open(filename1, "a", encoding="utf8") as output_file:
                output_file.writelines([cur_line + "\n"])

        # Return both rules and complexity like in stochastic builder
        return sorted(rule_2_relevance.items(), key=lambda x: x[1])[:top_k], self.complexity