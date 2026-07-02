import html
import os
from collections import defaultdict
import re
import numpy

from config import DATA_PATH



# relation types
ONE_TO_ONE="1-1"
ONE_TO_MANY="1-N"
MANY_TO_ONE="N-1"
MANY_TO_MANY="N-N"

VAR_PATTERN = re.compile(r'\B[?]{1}\w{1}\b') #'?' with single word character and word boundary before and after (" " or end/beginning of string)

class Dataset:

    def __init__(self,
                 name: str,
                 separator: str = "\t",
                 load: bool = True,
                 verbose: bool = True):
        """
            Dataset constructor.
            This method will initialize the Dataset and its structures.
            If parameter "load" is set to true, it will immediately read the dataset files
            and fill the data structures with the read data.
            :param name: the dataset name. It must correspond to the dataset folder name in DATA_PATH
            :param separator: the character that separates head, relation and tail in each triple in the dataset files
            :param load: boolean flag; if True, the dataset files must be accessed and read immediately.
        """

        # note: the "load" flag is necessary because the Kelpie datasets do not require loading,
        #       as they are built from already loaded pre-existing datasets.

        self.name = name
        self.separator = separator
        self.home = os.path.join(DATA_PATH, self.name)
        self.verbose = verbose

        # train, valid and test set paths
        self.train_path = os.path.join(self.home, "train.txt")
        self.valid_path = os.path.join(self.home, "valid.txt")
        self.test_path = os.path.join(self.home, "test.txt")

        ### All these data structures are read from filesystem lazily using the load method ###

        # sets of entities and relations in their textual format
        self.entities, self.relations = set(), set()

        # maps that associate to each entity/relation name (id) the corresponding entity/relation id (name)
        self.entity_name_2_id, self.entity_id_2_name = dict(), dict()
        self.relation_name_2_id, self.relation_id_2_name = dict(), dict()

        # collections of triples (facts with textual names) and samples (facts with numeric ids)
        self.train_triples, self.train_samples,\
        self.valid_triples, self.valid_samples, \
        self.test_triples, self.test_samples = None, None, None, None, None, None

        # Map each (head_id, rel_id) pair to the tail ids that complete the pair in train, valid or test samples.
        # This is used when computing ranks in filtered scenario.
        self.to_filter = defaultdict(lambda: list())

        # Map each (head_id, rel_id) pair to the tail ids that complete the pair in train samples only.
        # This is used by Loss functions that perform negative sampling.
        self.train_to_filter = defaultdict(lambda: list())

        # map each entity and relation id to its training degree
        self.entity_2_degree = defaultdict(lambda: 0)
        self.relation_2_degree = defaultdict(lambda: 0)

        # number of distinct entities and relations in this dataset.
        # Num_relations counts each relation twice, because the dataset adds an inverse fact for each direct one.
        # As a consequence, num_relations = 2*num_direct_relations
        self.num_entities, self.num_relations, self.num_direct_relations = -1, -1, -1

        if load:
            if not os.path.isdir(self.home):
                raise Exception("Folder %s does not exist" % self.home)

            # internal counter for the distinct entities and relations encountered so far
            self._entity_counter, self._relation_counter = 0, 0

            # read train, valid and test triples, and extract the corresponding samples; both are numpy arrays.
            # Triples feature entity and relation names; samples feature the corresponding ids.
            self.train_triples, self.train_samples = self._read_triples(self.train_path, self.separator)
            self.valid_triples, self.valid_samples = self._read_triples(self.valid_path, self.separator)
            self.test_triples, self.test_samples = self._read_triples(self.test_path, self.separator)

            # this is used for O(1) access to training samples
            self.train_samples_set = {(h, r, t) for (h, r, t) in self.train_samples}

            # update the overall number of distinct entities and distinct relations in the dataset
            self.num_entities = len(self.entities)
            self.num_direct_relations = len(self.relations)
            self.num_relations = 2*len(self.relations)  # also count inverse relations

            # add the inverse relations to the relation_id_2_name and relation_name_2_id data structures
            for relation_id in range(self.num_direct_relations):
                inverse_relation_id = relation_id + self.num_direct_relations
                inverse_relation_name = "INVERSE_" + self.relation_id_2_name[relation_id]
                self.relation_id_2_name[inverse_relation_id] = inverse_relation_name
                self.relation_name_2_id[inverse_relation_name] = inverse_relation_id

            # add the tail_id to the list of all tails seen completing (head_id, relation_id, ?)
            # and add the head_id to the list of all heads seen completing (?, relation_id, tail_id)
            all_samples = numpy.vstack((self.train_samples, self.valid_samples, self.test_samples))
            for i in range(all_samples.shape[0]):
                (head_id, relation_id, tail_id) = all_samples[i]
                self.to_filter[(head_id, relation_id)].append(tail_id)
                self.to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
                # if the sample was a training sample, also do the same for the train_to_filter data structure;
                # Also fill the entity_2_degree and relation_2_degree dicts.
                if i < len(self.train_samples):
                    self.train_to_filter[(head_id, relation_id)].append(tail_id)
                    self.train_to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
                    self.entity_2_degree[head_id] += 1
                    self.relation_2_degree[relation_id] += 1
                    if tail_id != head_id:
                        self.entity_2_degree[tail_id] += 1

            # map each relation id to its type (ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, or MANY_TO_MANY)
            self._compute_relation_2_type()

    def _read_triples(self, triples_path: str, separator="\t"):
        """
            Private method to read the triples (that is, facts and samples) from a textual file
            :param triples_path: the path of the file to read the triples from
            :param separator: the separator used in the file to read to separate head, relation and tail of each triple
            :return: a 2D numpy array containing the read facts,
                     and a 2D numpy array containing the corresponding samples
        """
        textual_triples = []
        data_triples = []
        
        with open(triples_path, "r", encoding="utf8" ) as triples_file:
            lines = triples_file.readlines()
            for line in lines:
                line = html.unescape(line).lower()   # this is required for some YAGO3-10 lines
                try:
                    head_name, relation_name, tail_name = line.strip().split(separator)
                except Exception as e:
                    if line == "###":
                        print(f"early exit marker '###' reached")   # print(*fact, sep = ", ")
                        break
                    if self.verbose: print(f"malformed line in 'triples to explain': '{line}'. {e}")
                    continue

                # remove unwanted characters
                head_name = self._remove_unwanted_characters(head_name)
                relation_name = self._remove_unwanted_characters(relation_name)
                tail_name = self._remove_unwanted_characters(tail_name)

                textual_triples.append((head_name, relation_name, tail_name))

                self.entities.add(head_name)
                self.entities.add(tail_name)
                self.relations.add(relation_name)

                if head_name in self.entity_name_2_id:
                    head_id = self.entity_name_2_id[head_name]
                else:
                    head_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[head_name] = head_id
                    self.entity_id_2_name[head_id] = head_name

                if relation_name in self.relation_name_2_id:
                    relation_id = self.relation_name_2_id[relation_name]
                else:
                    relation_id = self._relation_counter
                    self._relation_counter += 1
                    self.relation_name_2_id[relation_name] = relation_id
                    self.relation_id_2_name[relation_id] = relation_name

                if tail_name in self.entity_name_2_id:
                    tail_id = self.entity_name_2_id[tail_name]
                else:
                    tail_id = self._entity_counter
                    self._entity_counter += 1
                    self.entity_name_2_id[tail_name] = tail_id
                    self.entity_id_2_name[tail_id] = tail_name

                data_triples.append((head_id, relation_id, tail_id))

        return numpy.array(textual_triples), numpy.array(data_triples).astype('int64')

    @staticmethod
    def _remove_unwanted_characters(string:str)->str:
        #normal_string = re.sub("[^A-Za-z()_0-9]", "", string, 0)
        #return normal_string
        return string.replace(",", "").replace(":", "").replace(";", "")    # .replace(".", "")

    def invert_samples(self, samples: numpy.array):
        """
            This method computes and returns the inverted version of the passed samples.
            :param samples: the direct samples to invert, in the form of a numpy array
            :return: the corresponding inverse samples, in the form of a numpy array
        """
        output = numpy.copy(samples)

        output[:, 0] = output[:, 2]
        output[:, 2] = samples[:, 0]
        output[:, 1] += self.num_direct_relations

        return output


    def _compute_relation_2_type(self):
        """
            This method computes the type of each relation in the dataset based on the self.train_to_filter structure
            (that must have been already computed and populated).
            The mappings relation - relation type are written in the self.relation_2_type dict.
            :return: None
        """
        if len(self.train_to_filter) == 0:
            raise Exception("The dataset has not been loaded yet, so it is not possible to compute relation types yet.")

        relation_2_heads_nums = defaultdict(lambda: list())
        relation_2_tails_nums = defaultdict(lambda: list())

        for (x, relation) in self.train_to_filter:
            if relation >= self.num_direct_relations:
                relation_2_heads_nums[relation - self.num_direct_relations].append(len(self.to_filter[(x, relation)]))
            else:
                relation_2_tails_nums[relation].append(len(self.to_filter[(x, relation)]))

        self.relation_2_type = {}

        for relation in relation_2_heads_nums:
            average_heads_per_tail = numpy.average(relation_2_heads_nums[relation])
            average_tails_per_head = numpy.average(relation_2_tails_nums[relation])

            if average_heads_per_tail > 1.2 and average_tails_per_head > 1.2:
                self.relation_2_type[relation] = MANY_TO_MANY
            elif average_heads_per_tail > 1.2 and average_tails_per_head <= 1.2:
                self.relation_2_type[relation] = MANY_TO_ONE
            elif average_heads_per_tail <= 1.2 and average_tails_per_head > 1.2:
                self.relation_2_type[relation] = ONE_TO_MANY
            else:
                self.relation_2_type[relation] = ONE_TO_ONE

    def get_id_for_entity_name(self, entity_name: str):
        if VAR_PATTERN.match(entity_name):
            return entity_name

        entity_name = entity_name.lower()
        try:
            return self.entity_name_2_id[entity_name]
        except:
            try:
                name = entity_name.replace("(","\\(").replace(")","\\)").replace("?", ".") #"[\w]?")
                r = re.compile(name)
                l = [key for key in self.entity_name_2_id.keys() if r.match(key)]
                return self.entity_name_2_id[l[0]]
            except Exception as e:
                raise NameNotFoundException(f"entity '{name}' not in dict of entities: {e}. Entity not found in datas")
                

    def get_name_for_entity_id(self, entity_id: int):
        return self.entity_id_2_name[entity_id]

    def get_id_for_relation_name(self, relation_name: str):
        if VAR_PATTERN.match(relation_name):
            return relation_name
        #TODO: remove above, relations are no variables...
        relation_name = relation_name.lower()
        try:
            return self.relation_name_2_id[relation_name]
        except:
            try:
                name = relation_name.replace("?", ".").replace("(","\\(").replace(")","\\)")
                r = re.compile(name)
                l = [key for key in self.relation_name_2_id.keys() if r.match(key)]
                return self.relation_name_2_id(l[0])
            except Exception as e:
                raise NameNotFoundException(f"relation '{name}' not in dict of relations: {e}. check if rules file matches dataset")

    def get_name_for_relation_id(self, relation_id: int):
        return self.relation_id_2_name[relation_id]

    def add_training_samples(self, samples_to_add: numpy.array):
        """
            Add some samples to the training samples of this dataset.
            The to_filter and train_to_filter data structures will be updated accordingly
            :param samples_to_add: the list of samples to add, in the form of a numpy array
        """

        if len(samples_to_add) == 0:
            return

        self.train_samples = numpy.vstack((self.train_samples, samples_to_add))

        for (head, rel, tail) in samples_to_add:
            self.train_samples_set.add((head, rel, tail))
            self.to_filter[(head, rel)].append(tail)
            self.to_filter[(tail, rel + self.num_direct_relations)].append(head)
            self.train_to_filter[(head, rel)].append(tail)
            self.train_to_filter[(tail, rel + self.num_direct_relations)].append(head)

    def sample_to_fact(self, sample_to_convert):
    # Handle string input
     if isinstance(sample_to_convert, str):
        # Check if it's a single word (likely an entity or relation name)
        str_parts = sample_to_convert.strip().split()
        if len(str_parts) == 1:
            # Single entity/relation name - just return it as is
            return sample_to_convert.strip()
        elif len(str_parts) == 3:
            # Single triple as string "head relation tail"
            return tuple(str_parts)
        else:
            # Try to convert using string_to_samples for multiple triples
            try:
                samples = self.string_to_samples(sample_to_convert)
                if len(samples) == 1:
                    # Single sample case
                    head_id, rel_id, tail_id = samples[0]
                    return self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
                else:
                    # Multiple samples case
                    result = []
                    for sample in samples:
                        head_id, rel_id, tail_id = sample
                        triple_tuple = self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
                        result.append(", ".join(triple_tuple))
                    return result
            except (TripleException, NameNotFoundException):
                # If conversion fails, it might be a single entity/relation name
                return sample_to_convert.strip()
    
    # Handle numpy arrays
     if isinstance(sample_to_convert, numpy.ndarray):
        if sample_to_convert.ndim == 1 and len(sample_to_convert) == 3:
            # Single sample as numpy array
            head_id, rel_id, tail_id = sample_to_convert
            return self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
        elif sample_to_convert.ndim == 2:
            # Multiple samples as 2D numpy array
            result = []
            for sample in sample_to_convert:
                head_id, rel_id, tail_id = sample
                triple_tuple = self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
                result.append(", ".join(triple_tuple))
            return result
    
    # Check if it's a single triple (tuple/list of 3 integers)
     if (hasattr(sample_to_convert, '__len__') and len(sample_to_convert) == 3 and 
        all(isinstance(x, (int, numpy.int32, numpy.int64)) for x in sample_to_convert)):
        head_id, rel_id, tail_id = sample_to_convert
        return self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
    
    # Handle list/tuple of triples
     if hasattr(sample_to_convert, '__iter__'):
        result = []
        for triple in sample_to_convert:
            # Check if triple is a string (shouldn't happen at this level)
            if isinstance(triple, str):
                # If it's a single word, just add it as is
                if len(triple.strip().split()) == 1:
                    result.append(triple.strip())
                    continue
                else:
                    raise ValueError(f"Expected tuple with 3 integers, got string: '{triple}'")
            
            # Add validation to ensure each triple has exactly 3 elements
            if not hasattr(triple, '__len__') or len(triple) != 3:
                raise ValueError(f"Expected triple with 3 elements, got {len(triple) if hasattr(triple, '__len__') else 'non-sequence'} elements: {triple}")
            
            head_id, rel_id, tail_id = triple
            triple_tuple = self.entity_id_2_name[head_id], self.relation_id_2_name[rel_id], self.entity_id_2_name[tail_id]
            result.append(", ".join(triple_tuple))
        return result
    
    # If we get here, the input type is not supported
     raise ValueError(f"Unsupported input type: {type(sample_to_convert)}, value: {sample_to_convert}")

    def fact_to_sample(self, fact_to_convert: tuple[str, str, str], type:str=None):
        if len(fact_to_convert) != 3:
            raise TripleException(f"fact_to_convert does not have length 3: {fact_to_convert}")

        head_name, rel_name, tail_name = fact_to_convert
        if type == "str":
            return str(self.get_id_for_entity_name(head_name.strip())), str(self.get_id_for_relation_name(rel_name.strip())), str(self.get_id_for_entity_name(tail_name.strip()))
        else:
            return self.get_id_for_entity_name(head_name.strip()), self.get_id_for_relation_name(rel_name.strip()), self.get_id_for_entity_name(tail_name.strip())

    def string_to_samples(self, string_to_convert: str, type=None) -> list[tuple[str, str, str]] | None:
        str_list = string_to_convert.split(" ")
        str_list = list(filter(None, str_list))
        if len(str_list) == 3:
            return [self.fact_to_sample(str_list, type)]
        elif len(str_list) % 3 == 0:
            i = 0
            facts = []
            while i < len(str_list):
                facts.append(self.fact_to_sample((str_list[i], str_list[i+1], str_list[i+2]), type))
                i += 3
            return facts
        else:
            raise TripleException(f"string '{str_list}' not made of triples.")

    def remove_training_samples(self, samples_to_remove: numpy.array):
        """
        This method quietly removes a bunch of samples from the training set of this dataset.
        If asked to remove samples that are not present in the training set, this method does nothing and returns False.
        :param samples_to_remove: the samples to remove as a 2D numpy array
                                  in which each row corresponds to one sample to remove

        :return: a 1D array of boolean values as long as passed array of samples to remove;
                 in each position i, it contains True if the i-th passed sample to remove
                 was actually included in the training set, so it was possible to remove it; False otherwise.
        """

        indices_to_remove = []
        removed_samples = []
        output = []
        for sample_to_remove in samples_to_remove:
            head, rel, tail = sample_to_remove

            if (head, rel, tail) in self.train_samples_set:
                index = numpy.where(numpy.all(self.train_samples == sample_to_remove, axis=1))[0]
                if len(index)>1:
                    for i in index:
                        indices_to_remove.append(numpy.array([i], dtype=numpy.int64))     #for duplicate triples
                else: 
                    indices_to_remove.append(index)
                removed_samples.append(self.train_samples[index])

                self.to_filter[(head, rel)].remove(tail)
                self.to_filter[(tail, rel + self.num_direct_relations)].remove(head)
                self.train_samples_set.remove((head, rel, tail))
                output.append(True)
            else:
                output.append(False)

        self.train_samples = numpy.delete(self.train_samples, indices_to_remove, axis=0)
        return output

    def remove_training_sample(self, sample_to_remove: numpy.array):
        """
        This method quietly removes a sample from the training set of this dataset.
        If asked to remove a sample that is not present in the training set, this method does nothing and returns False.
        :param sample_to_remove: the sample to remove

        :return: True if the passed sample was actually included in the training set, so it was possible to remove it;
                 False otherwise.
        """

        head, rel, tail = sample_to_remove

        if (head, rel, tail) in self.train_samples_set:
            index = numpy.where(numpy.all(self.train_samples == sample_to_remove, axis=1))
            self.train_samples = numpy.delete(self.train_samples, index[0], axis=0)

            self.to_filter[(head, rel)].remove(tail)
            self.to_filter[(tail, rel + self.num_direct_relations)].remove(head)
            self.train_samples_set.remove((head, rel, tail))
            return True
        return False

    @staticmethod
    def replace_entity_in_sample(sample, old_entity: int, new_entity:int, as_numpy=True):
        h, r, t = sample
        if h == old_entity:
            h = new_entity
        if t == old_entity:
            t = new_entity
        return numpy.array([h, r, t]) if as_numpy else (h, r, t)

    @staticmethod
    def replace_entity_in_samples(samples:list[list[int, int, int]], old_entity: int, new_entity:int, as_numpy=True):
        result = []
        for (h, r, t) in samples:
            if h == old_entity:
                h = new_entity
            if t == old_entity:
                t = new_entity
            result.append((h, r, t))

        return numpy.array(result) if as_numpy else result

    def printable_sample(self, sample: tuple[int, int, int]):
        return "<" + ", ".join(self.sample_to_fact(sample)) + ">"

    def printable_nple(self, nple: list):
        return" + ".join([self.printable_sample(sample) for sample in nple])
    
class NameNotFoundException(Exception):
    pass

class TripleException(Exception):
    pass