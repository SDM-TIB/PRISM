import ast
from collections import defaultdict
import copy
import itertools
import operator
import os
import random
import re
import sys
import duckdb
import numpy
import pandas as pd
import rdflib
from rdflib import Graph, URIRef
from rdflib.plugins.sparql import prepareQuery
import traceback

from config import RULES_PATH
from src.model.utils import flatten_list
from src.model.dataset import Dataset, NameNotFoundException
from src.relevance_engines.post_training_engine import PostTrainingEngine

DEAFAULT_XSI_THRESHOLD = 5
IMPROVEMENT_THRESHOLD = 0
ABSURDLY_LOW_VALUE = -1e6
ARTIFICAL_PCA = 0.99

class Ruleset:

    def __init__(self,
                 dataset: Dataset,
                 rules_file1: str,
                 rules_file2: str = None,
                 pca_threshold:float = 0.0,
                 order_col = 'pca_confidence',
                 load: bool = True,
                 sep:str ='\t',
                 decimal:str = '.',
                 verbose: bool = True):
        """
            Ruleset constructor.
            This method will initialize the Ruleset and its structures.
            If parameter "load" is set to true, it will immediately read the dataset files and fill the data structures with the read data.

        Args:
            rules_file (str): _description_
            rules_file2 (str): _description_
            pca_threshold (float, optional): _description_. Defaults to 0.7.
            order_col (str, optional): _description_. Defaults to 'pca_confidence'.
            load (bool, optional): boolean flag; if True, the dataset files must be accessed and read immediately.. Defaults to True.
            sep (str, optional): the character that separates columns in the rules files. Defaults to ','.
            decimal (str, optional): _description_. Defaults to '.'.
            verbose (bool, optional): _description_. Defaults to True.

        Raises:
            Exception: _description_
        """        

        self.dataset = dataset
        self.order_col:str = order_col
        self.verbose:bool = verbose
        self.rules_df_names:pd.DataFrame = None
        self.rules_df_ids:pd.DataFrame = None
        self.rules_path_1 = os.path.join(RULES_PATH, rules_file1) 
        if rules_file2 != None:
            self.rules_path_2 = os.path.join(RULES_PATH, rules_file2)
        else:
            self.rules_path_2 = None
    
        self.sufficient_row_ids = {}
        self.necessary_row_ids = {}
        self.sufficient_pca = {}
        self.necessary_pca = {}
        self.full_graph = None
        self.full_graph = self._load_graph()
        
        if load:
            if not os.path.isfile(self.rules_path_1):
                raise Exception("File %s does not exist" % self.rules_path_1)
            
            # Rule import        #TODO: cover different formats or use auto format detection?
            rules_df1 = pd.read_csv(self.rules_path_1, sep=sep, decimal=decimal, header=0, encoding='utf8')  # load rules
            rules_df1.columns = rules_df1.columns.str.replace(' ', '_')
            rules_df1.columns = map(str.lower, rules_df1.columns)
            if verbose: print(f"loading Rules...   {self.rules_path_1}")

            if self.rules_path_2 != None:
                if os.path.isfile(self.rules_path_2):
                    if verbose: print(f"joining editorial Rules...   {self.rules_path_2}")
                    rules_df2 = pd.read_csv(self.rules_path_2, decimal=decimal, header=0)  # load editorial rules
                    rules_df2.columns = rules_df2.columns.str.replace(' ', '_')
                    rules_df2.columns = map(str.lower, rules_df2.columns)
                    self.rules_df_names = pd.concat([rules_df1,rules_df2], axis=0, ignore_index=True).fillna(0.99)
                else:
                    if verbose: print("rulefile 2 path not valid: {self.rules_path_2}. Skipping editorial rules file")
                    self.rules_df_names = rules_df1
            else:
                if verbose: print("no editorial Rules")
                self.rules_df_names = rules_df1


            # filter and reduce rules_df 
            self.rules_df_names = self.rules_df_names.loc[self.rules_df_names[self.order_col] > pca_threshold]
            #self.rules_df_names = self.rules_df_names.loc[self.rules_df_names[self.order_col] != 1]
            self.rules_df_names = self.rules_df_names[['body', 'head', self.order_col, 'functional_variable']]

            # different file format with rule col, instead of body&head
            if "Rule" in self.rules_df_names.columns:
                self.rules_df_names[['body', 'head']] = self.rules_df_names['Rule'].apply(lambda x: pd.Series(self._split_hornrule_into_head_body(x)))  # split rule into body and head triples
                self.rules_df_names = self.rules_df_names.drop('Rule', axis=1)   # remove original rule string

            self.rules_df_names["head"] = self._remove_unwanted_characters(self.rules_df_names["head"])
            self.rules_df_names["body"] = self._remove_unwanted_characters(self.rules_df_names["body"])

            self.rules_df_names.reset_index(drop=True, inplace=True)

            ## transform names to ids
            self.rules_df_ids = self.rules_df_names.copy()
            self.rules_df_ids['head'] = self.rules_df_names['head'].apply(lambda x: self.sample_str_or_none(x)) # for duckdb this col needs to be a str
            self.rules_df_ids['body'] = self.rules_df_names['body'].apply(lambda x: self.sample_str_or_none(x))
            self.rules_df_ids.dropna()

    def sample_str_or_none(self, string):
        try:
            _list = self.dataset.string_to_samples(string, type="str")
            string_list = []
            for l in _list:
                string_list.append(" ".join(l))
            return ". ".join(string_list)
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def _remove_unwanted_characters(df_col):
        # same as in dataset.py line 163
        col1 = df_col.str.replace(",", "")
        col1 = col1.str.replace(":", "")
        col1 = col1.str.replace(";", "")
        #col1 = col1.str.replace(".", "")
        return col1


    def find_rules_and_evidence(self, relation_id:str, target_id:str, start_entity_id:int, top_k:int)->  tuple[list[list[int, int, int]], list[list[int, int, int]]]:
        # This function finds all rules evident in the (ego)graph around the prediction (incl. containing constants). 
        # It returns a list of triples of instanciated rules, grouped by rule (for nec)

        rules_df = self.rules_df_ids
        relation = str(relation_id)
        target = str(target_id)
        start_entity = str(start_entity_id)

        if rules_df['head'].str.contains(target).any():     # if target entity is contained in rule_df -> rules with constants
            q1 = f"""SELECT * FROM rules_df WHERE head LIKE '% {relation} %{target}%' OR head LIKE '% {relation} %?%' ORDER BY {self.order_col} DESC"""     # include rules with correct constant and variable as target
        else:
            q1 = f"""SELECT * FROM rules_df WHERE head LIKE '% {relation} %' ORDER BY {self.order_col} DESC"""

        relevant_rules = duckdb.query(q1).df()
        
        if relevant_rules.empty:
            raise EmptyResultException(f"no rules applicable to the relation '{relation}'")
        
        if self.verbose:
            print(q1)
            print("relevant_rules:")
            print(relevant_rules)
            print()

        # ego graph is a subgraph focused on the start_entity. It is created once for each prediction and then used to find evidence both for rules and "same-realtion" rule 
        ego_graph = self._load_ego_graph(start_entity_id, self._longest_rule_length(relevant_rules)) 
        
        rules_evidence = self._find_evidence_via_ids(relevant_rules, start_entity_id, ego_graph) 

        # in addition to rules: use similarity of predicted relation
        relation_rule = f":{start_entity} :{relation} ?b"
        qres:list[rdflib.query.ResultRow] = self._query_graph_for_evidence(["?b"], relation_rule, ego_graph)
        if qres == None:
            return rules_evidence

        relation_list = []
        # if qres == None:
        #     return rules_evidence
        for items in qres:
            for item in items:
                rule_tuple = ((start_entity_id, relation_id, int(item)),)
                relation_list.append(rule_tuple)
                print("added: ", rule_tuple)

                self.sufficient_pca[tuple(rule_tuple)] = ARTIFICAL_PCA
                self.necessary_pca[tuple(rule_tuple)] = ARTIFICAL_PCA
        
        rules_evidence = (rules_evidence[0] + relation_list, rules_evidence[1] + relation_list)
        return rules_evidence
    
    @staticmethod
    def _longest_rule_length(relevant_rules:pd.DataFrame):
        entry_count = relevant_rules['body'].str.count(' ').max() + 1
        # 2 spaces = 3 entries = 1 triple
        # 5 spaces = 6 entries = 2 triples

        if (entry_count % 3) != 0:
            raise Exception(f"counted {max} entries, but number of entries must be divisible by 3")
        

        return  int(entry_count/3)
        

    def _load_graph(self) -> Graph:
        if self.full_graph != None:
            return self.full_graph

        g1 = Graph()
        for line in self.dataset.train_samples:
            #triple = [(prefix + t) for t in triple]
            triple = (URIRef(str(t)) for t in line) # we have to wrap them in URIRef
            g1.add(triple)                    # and add to the graph
            # Successfully parsed this line

        return g1
    
    def _load_ego_graph(self, central_entity: int = None, hops: int = None) -> Graph:
        # TODO: remove after debugging ego_graph
        return self.full_graph

        g_full = self.full_graph

        # If no central entity or hops is provided, return the full graph
        if central_entity is None or hops is None:
            return g_full
        try:
            g_ego = Graph()

            # Convert central_entity to URIRef
            central_entity_uri = URIRef(str(central_entity))

            # SPARQL Construct query to get nodes within number hops from the central entity
            query_str = f"""
                PREFIX : <>
                CONSTRUCT {{
                    ?s ?p ?o .
                }}
                WHERE {{
                    # out
                    {{
                        <{central_entity_uri}> ?p1 ?o1 .
                        BIND(<{central_entity_uri}> AS ?s)
                        BIND(?p1 AS ?p)
                        BIND(?o1 AS ?o)
                    }}
                    UNION
                    # in
                    {{
                        ?s1 ?p1 <{central_entity_uri}> .
                        BIND(?s1 AS ?s)
                        BIND(?p1 AS ?p)
                        BIND(<{central_entity_uri}> AS ?o)
                    }}
                """

            if hops > 1:
                query_str += f"""
                UNION
                # out out
                {{
                    <{central_entity_uri}> ?p1 ?o1 .
                    ?o1 ?p2 ?o2 .
                    BIND(?o1 AS ?s)
                    BIND(?p2 AS ?p)
                    BIND(?o2 AS ?o)
                }}
                UNION
                # in in
                {{
                    ?s1 ?p1 <{central_entity_uri}> .
                    ?s2 ?p2 ?s1 .
                    BIND(?s2 AS ?s)
                    BIND(?p2 AS ?p)
                    BIND(?s1 AS ?o)
                }}
                UNION
                # out in
                {{
                    <{central_entity_uri}> ?p1 ?o1 .
                    ?o2 ?p2 ?o1 .
                    BIND(?o2 AS ?s)
                    BIND(?p2 AS ?p)
                    BIND(?o1 AS ?o)
            }}
                UNION
                # in out
                {{
                    ?s1 ?p1 <{central_entity_uri}> .
                    ?s1 ?p2 ?s2 .
                    BIND(?s1 AS ?s)
                    BIND(?p2 AS ?p)
                    BIND(?s2 AS ?o)
                }}
                """
                
                if hops > 2:
                    query_str += f"""
                        UNION
                        # out out out
                        {{
                            <{central_entity_uri}> ?p1 ?o1 .
                            ?o1 ?p2 ?o2 .
                            ?o2 ?p3 ?o3 .
                            BIND(?o2 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?o3 AS ?o)
                        }}
                        UNION
                        # out out in
                        {{
                            <{central_entity_uri}> ?p1 ?o1 .
                            ?o1 ?p2 ?o2 .
                            ?o3 ?p3 ?o2 .
                            BIND(?o3 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?o2 AS ?o)
                        }}
                        UNION
                        # in in out
                        {{
                            ?s1 ?p1 <{central_entity_uri}> .
                            ?s2 ?p2 ?s1 .
                            ?s2 ?p3 ?s3 .
                            BIND(?s2 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?s3 AS ?o)
                        }}
                        UNION
                        # in in in
                        {{
                            ?s1 ?p1 <{central_entity_uri}> .
                            ?s2 ?p2 ?s1 .
                            ?s3 ?p3 ?s2 .
                            BIND(?s3 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?s2 AS ?o)
                        }}
                        UNION
                        # out in out
                        {{
                            <{central_entity_uri}> ?p1 ?o1 .
                            ?o2 ?p2 ?o1 .
                            ?o2 ?p3 ?o3 .
                            BIND(?o2 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?o3 AS ?o)
                        }}
                        UNION
                        # out in in
                        {{
                            <{central_entity_uri}> ?p1 ?o1 .
                            ?o2 ?p2 ?o1 .
                            ?o3 ?p3 ?o2 .
                            BIND(?o3 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?o2 AS ?o)
                        }}
                        UNION
                        # in out out
                        {{
                            ?s1 ?p1 <{central_entity_uri}> .
                            ?s1 ?p2 ?s2 .
                            ?s2 ?p3 ?s3 .
                            BIND(?s2 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?s3 AS ?o)
                        }}
                        UNION
                        # in out in
                        {{
                            ?s1 ?p1 <{central_entity_uri}> .
                            ?s1 ?p2 ?s2 .
                            ?s3 ?p3 ?s2 .
                            BIND(?s3 AS ?s)
                            BIND(?p3 AS ?p)
                            BIND(?s2 AS ?o)
                        }}
                """

                    if hops > 3:
                        raise exception("more than 3 hops are solved globaly, not locally")
                    
                query_str += "}"
            print(query_str)        #TODO remove after debugging

            # Prepare and execute the query
            query = prepareQuery(query_str)
            results = g_full.query(query)

            # Adding results to the ego graph
            for triple in results:
                g_ego.add(triple)
        except Exception as e:
            print(f"no ego graph possible for central entity '{central_entity}', using {hops} hops:")
            print(e)
            print(traceback.format_exc())
            return g_full

        return g_ego
    
    
    def _find_evidence_via_ids(self, relevant_rule_df:pd.DataFrame, entity_id:int, ego_graph=None)  \
        -> tuple[list[list[int, int, int]], list[list[int, int, int]]]:
        """_summary_

        Args:
            rule_df (_type_): _description_
            entity_id (_type_): _description_
            top_k (_type_): _description_
            ego_graph (_type_, optional): _description_. Defaults to None.

        Returns:
            list[tuple[list[str]]]: a list of rule tuples, each triple in a rule is represented as a list of strings. For each rule #TODO more than one result per rule...
        """

        self.nec_result = [] # sufficient: we need each instanciation of a rule in seperate list, to add single instaces/chains to convert
        self.suf_result = [] # necessary we need a single list of all instanciations of a rule, to remove at once to reduce score

        entity_name = self.dataset.entity_id_2_name[entity_id]
        self.num_samples_found = 0

        for idx, item in relevant_rule_df.iterrows():    # for each applicable rule
            #if self.num_samples_found > top_k:
            #    break

            fun_var:str = item['functional_variable']
            body:str = item['body']
            head:str = item['head']

            if body == None:
                print(f"------- BODY NONE : entity {entity_name}, head: {head}, var: {fun_var}------")
                break


            # Replace functional variable in body with central entity_id
            body = body.replace(fun_var, str(entity_id))   # id bc id-rules-df # TODO quicker if not name but id, but it complicates line 161ff ... "replace relatios & entities by id's"


            try:
                variables, rule_strings, triples = self._extract_vars_and_rules(body)
            except NameNotFoundException as e:
                # if this rule can not be properly transformed (str -> int) then skip this rule
                print(e)    #TODO: check all rules at beginning and remove such rules.
                continue

            qres:list[rdflib.query.ResultRow] = self._query_graph_for_evidence(variables, rule_strings, ego_graph)
            if qres == None:
                continue
            if self.verbose: print(f"Rule: {body} -> {head}")

            suf, nec = self._extract_sets(qres, triples)

            # store row id for each suf & nec set, to later identify eg. pca confidence in self.dataFrame
            row_id = self.extract_row_id(item['body'], item['head'])
            pca = self.extract_PCA(item['body'], item['head'])

            #suf_tuple = tuple(suf) if len(suf)>1 else suf[0]
            #self.sufficient_row_ids[tuple(suf)] = row_id
            self.sufficient_row_ids[tuple(nec)] = row_id #TODO: [single suf] use tuple(suf) for single removal
            self.necessary_row_ids[tuple(nec)] = row_id
            #self.sufficient_pca[suf_tuple] = pc
            self.sufficient_pca[tuple(nec)] = pca #TODO: [single suf] use tuple(suf) for single removal
            self.necessary_pca[tuple(nec)] = pca

        #if self.verbose: print("suf: ", self.suf_result)
        if self.verbose: print("nec: ", self.nec_result)

        if len(self.nec_result) == 0:
            raise EmptyResultException("no rule applies to this entity, necessary and sufficient sets are empty")

        return self.nec_result, self.nec_result #TODO: [single suf] return suf_result for single removal
    
    def extract_row_id(self, body, head):
        row_id = self.rules_df_names.index[
            ((self.rules_df_ids['body'] == body) & 
             (self.rules_df_ids['head'] == head))
             ].tolist()
        
        if len(row_id) < 1:
            raise Exception(f"no rule matches this body and head:\nlooking for: {body} -> {head}\ndf: {self.rules_df_names}")
        elif len(row_id) > 1:
            raise Exception(f"more than one rule matches this body and head:\nlooking for: {body} -> {head}\nrows: {row_id}: {self.rules_df_names.iloc[row_id]}")
        
        return row_id[0]
    
            
    def extract_PCA(self, body, head):
        pca = self.rules_df_names.pca_confidence[
            ((self.rules_df_ids['body'] == body) & 
             (self.rules_df_ids['head'] == head))
             ].tolist()
        
        if len(pca) < 1:
            raise Exception(f"no rule matches this body and head:\nlooking for: {body} -> {head}\ndf: {self.rules_df_names}")
        elif len(pca) > 1:
            row_id = self.extract_row_id(body, head)
            raise Exception(f"more than one rule matches this body and head:\nlooking for: {body} -> {head}\nrows: {row_id}: {self.rules_df_names.iloc[row_id]}")
        
        return pca[0]
            
    def _extract_vars_and_rules(self, body:str)-> tuple[list[str], list[str], list[list[str]]]:
        
        # identify variables like ?b for SPARQL query
        var_pattern = re.compile(r'\B[?]{1}\w{1}\b') #'?' with single word character and word boundary before and after (" " or end/beginning of string)
        variables:list[str] = var_pattern.findall(body)
        variables:set[str] = set(variables)

        rule = []
        triples = []

        list_o_triples = []
        if "." in body:
            list_o_triples = body.split(".")
        else:
            list_o_triples = [body]
        
        for i, triple in enumerate(list_o_triples):
            triple = triple.split(" ")
            triple = list(filter(None, triple))
            triples.append(copy.deepcopy(triple))  # triple list for result
            for j,t in enumerate(triple):
                if re.match(var_pattern, t) == None:    # no variable
                    triple[j] = ":"+t
            rule.append(" ".join(triple)) 
        rule_string = ". ".join(rule)
            
        #TODO: add ":" when writing body in df, not here!

        # result samples:
        # triples = [['61231', '0', '?b'], ['35559', '0', '?b']]
        # rule_string = ':61231 :0 ?b. :35559 :0 ?b. '

        return variables, rule_string, triples


    def _query_graph_for_evidence(self, variables, rule_string, ego_graph=None)-> list[rdflib.query.ResultRow]:

        #TODO: only load ego graph?
        graph = ego_graph if ego_graph != None else self.full_graph

        # if no variables, ask query
        if len(variables) == 0:
            query = f"PREFIX : <> ASK  {{{rule_string}}}"
            #if self.verbose: print(query)

            qres = graph.query(query)
            if qres.askAnswer == True:
                qres = qres
            # if query result is empty, continue search
            elif qres.askAnswer == False:
                return None
            else:
                print("error reading qres boolean. ASK Query might have failed")

        # construct query from all variables and parts of rule body
        else:
            ask_query = f"PREFIX : <> ASK {{{rule_string}}}"
            qres = graph.query(ask_query)
            for row in qres:
                if row == True:     # if rule does not apply to this entity, skip this rule and continue search
                    query = f"PREFIX : <> SELECT DISTINCT "
                    for var in variables:
                        query = query + var + " "
                    query = f"{query} WHERE {{{rule_string}}}"
                    print(query)        #TODO remove after debugging
                    qres = graph.query(query)            
                    # if query result is empty, skip this rule and continue search
                    if len(qres) == 0:
                        return None
                    else:
                        self.num_samples_found += 1
                else:   # Ask query only has a single row
                    return None
        return qres
    
    def _extract_sets(self, qres, rule_triples)-> tuple[list, list]:
        nec = []
        suf = []

        #TODO: avoid these loops by using different query return format?

        # convert triples with variables into triples from Graph
        for items in qres:  # for each row in result = for each instanciation of rule
            instance = copy.deepcopy(rule_triples)
            new_triples = []
            for i, triple in enumerate(instance):    # check all triples
                if type(items) == bool:
                    items = [""]
                    var = [""]
                else:
                    var = iter(items.labels)
                for item, v in zip(items, var):   # for each variable in query
                    v = f"?{v}"
                    for j, t in enumerate(triple):
                        t_new = t
                        if type(t) == str:
                            t_new = t.replace(v, item) # replace variable with found entity
                            try:    # cast to int if not a var but an entity
                                t_new = int(t_new)
                            except:
                                pass
                        instance[i][j] = t_new
                new_triples.append(tuple(triple))
            self.suf_result.append(tuple(new_triples))  # suf grouped by instanciation #TODO update tuple type
            suf.append(tuple(new_triples)) #suf set return (for storing with row id)
            nec += new_triples
        self.nec_result.append(tuple(nec)) # nec grouped by rule
        print(nec)
        return suf, nec
    

    def _split_hornrule_into_head_body(self, rule:str)->tuple[list[str], list[str]]:
        # Split the rule into parts separated by '=>'
        left, right = rule.split('=>')
        body_triples = self._extract_statements(left)
        head_triples = self._extract_statements(right)
        
        return body_triples, head_triples
    
        
    def _extract_statements(self, triple_string:str)->list[str]:
        triple_string_clean = re.sub(r"\s+", ' ', triple_string)
        # Extract statements from longer string
        string_list = re.findall(r'[^\s]+ [^\s]+ [^\s]+', triple_string_clean)
        return string_list

    #method for shorter print
    @staticmethod
    def extract_relations(rules:list[list[tuple[int, int, int]]]|list[tuple[int, int, int]]|tuple[int, int, int], dataset) -> list[str]|list[list[str]]:
        if (type(rules) == tuple) & (type(rules[0]) == int):    # single triple
            return dataset.sample_to_fact(rules)
        
        relation_list = []
        for i, group in enumerate(rules):
            if (len(group) == 3) & (type(group[0]) == int):     # group:tuple[int, int, int]
                sample = dataset.sample_to_fact(group)          # sample:tuple[str, str, str]
                relation_list.append(sample[1])                 # only append predicate
            elif (len(group[0]) == 3) & (type(group[0][0]) == int):
                relation_list.append([])
                for tup in group:               # tuple:tuple[int, int, int]
                    sample = dataset.sample_to_fact(tup)      # sample:tuple[str, str, str]
                    relation_list[i].append(sample[1])          # only append predicate
            else:
                relation_list.append(f"malformed group '{group}'")
        return relation_list


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
    

    def printable_nple(self, nple: list):
        return " + ".join([self.printable_sample(sample) for sample in nple])

    def printable_sample(self, sample: tuple[int, int, int]):
        return "<" + ", ".join(self.dataset.sample_to_fact(sample)) + ">"

class RuleEvidence:
    def __init__(self, 
                 ruleset: Ruleset,
                 dataset: Dataset,
                 engine:PostTrainingEngine,
                 perspective: str,
                 mode:str,
                 sample_to_explain:tuple[int, int, int],
                 output_prefix:str,
                 perspective_entity:int,
                 relevance_threshold: float = None,
                 entities_to_convert: list = None,
                 verbose:bool = True):
        
        self.perspective_entity = perspective_entity
        self.mode = mode
        self.verbose = verbose
        self.ruleset:Ruleset = ruleset
        self.dataset:Dataset = dataset
        self.engine = engine
        self.perspective = perspective
        self.sample_to_explain = sample_to_explain
        self.output_prefix = output_prefix
        self.entities_to_convert = entities_to_convert

        self.xsi = relevance_threshold if relevance_threshold is not None else DEAFAULT_XSI_THRESHOLD
        self.window_size = 10
        self.improvement_threshhold = IMPROVEMENT_THRESHOLD

        self.complexity:dict = {}
        self.sample_2_relevance:dict = {}
        self.prelim_rule_score:dict = {}


    ### getter - setter ###


    def set_rules_to_remove(self, rules_to_remove: list[list[list[int, int, int]]], top_k)->None:
        self.rules_to_remove = rules_to_remove
        relevant_triples_list = flatten_list(rules_to_remove) # turn list of rules into list of triples, with duplicates
        self.relevant_triples_set = tuple(set(relevant_triples_list)) # turn list of triples into tuple of triples, without duplicates
        self.top_k = top_k
    
    def get_complexity(self):
        return self.complexity


    ### ---- ###


    def extract_sample_relevance(self, samples_to_remove: tuple[tuple[int, int, int]]):

        sample_2_preliminary_relevance = {}
        retrainings = 0

        # all triples are tested for preliminary score
        for i, sample_to_remove in enumerate(samples_to_remove):
            if self.verbose: print("\nComputing relevance for triple " + str(i+1) + " on " + str(
                len(samples_to_remove)) + ": " + self.dataset.printable_sample(sample_to_remove))
            triple_relevance = self._compute_relevance_for_rule(([sample_to_remove]), rule_length=0)
            retrainings +=1
            sample_2_preliminary_relevance[sample_to_remove] = triple_relevance
            if self.verbose: print("\tObtained relevance: " + str(triple_relevance))
            """if self.max_retrainings != None:
                if retrainings >= self.max_retrainings:
                    print(f"number of retrainings is at {retrainings}, max of {self.max_retrainings} reached, aborting further search. not counting into total")
                    #self.complexity["max_retrain"] += 1 # not counted here into sum, because it will for sure early exit again in the recombinations #TODO: remove
                    break"""
        self.sample_2_relevance = sample_2_preliminary_relevance
        return sample_2_preliminary_relevance
    
    
    def extract_set_relevance(self, rule_set:list[tuple[int, int, int]]|list[list[tuple[int, int, int]]])->float:
        if type(rule_set[0]) == int:
            raise Exception("should be list of triples")
        if type(rule_set[0][0]) != int:
            rule_set = flatten_list(rule_set)
        relevance = self._compute_relevance_for_rule(rule_set, len(rule_set))
        if self.verbose: print(f"{relevance}: {rule_set}")
        return relevance
    
    
    def extract_partial_set_relevance(self, full_set_relevance:float)->list:
        full_set = tuple(set(flatten_list(self.rules_to_remove)))
        necessary_set = []
        leng = len(self.rules_to_remove) - 1
        confidence_dict = self.extract_rule_confidence(rules=self.rules_to_remove)
        # sorting first by length, so shorter rules arent ignored because they are contained in longer ones
        sorted_rule_confidence = sorted(confidence_dict.items(), key=lambda x: (-len(x[0]), x[1]), reverse=True)

        # add single rules back to graph, if relevance of removed set decreases, this rule is important & added to necessary set
        trainings = 0
        for full_rule, rule_conficendce in sorted_rule_confidence:
            full_rule = tuple(set(full_rule))    # in case of duplicates in data/rules  #TODO: check again
            partial_set = list(copy.deepcopy(full_set))
            for triple in full_rule:
                try:
                    partial_set.remove(triple)  # possibly this triple was removed before
                except:
                    pass
            if partial_set == full_set:
                continue    # if nothing was removed, skip rule
            if len(partial_set) == 0:
                break       # if there is nothing more that can be removed, finish loop
            partial_relevance = self._compute_relevance_for_rule(partial_set, leng)
            trainings += 1
            
            if full_set_relevance > partial_relevance:      # if partial relevance is lower, this rule is important
                necessary_set.append(list(full_rule))
            else:
                print("  >skipped)")
            print(f"pca: {round(rule_conficendce, 3)} | Relavance with {self.short_print_rule_relevance({full_rule:partial_relevance}, inplace=False)}")
        if len(necessary_set) == 0:     # if none of the parts are important by themselves, then use all together   
            necessary_set = full_set
        return necessary_set
        

    def extract_rule_relevance(self):

        # combine preliminary triple score into score per rule
        all_rules_with_preliminary_scores = [
            (tuple(x), self._calc_preliminary_rule_score(x)) 
            for x in self.rules_to_remove]
        
        # sort rules by preliminary score
        all_rules_with_preliminary_scores = sorted(all_rules_with_preliminary_scores,
                                                            key=lambda x: x[1], reverse=True)

        rule_2_relevance_dict = {}

        terminate = False
        best_relevance_so_far = ABSURDLY_LOW_VALUE

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        while i < len(all_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_rules_with_preliminary_scores[i]

            # compute relevance for all single rules
            if self.verbose: print("\n\tComputing relevance for rule: " + self.dataset.printable_nple(current_rule))
            current_rule_relevance = self._compute_relevance_for_rule(current_rule, rule_length=1)
            rule_2_relevance_dict[current_rule] = current_rule_relevance
            if self.verbose: print("\tObtained relevance: " + str(current_rule_relevance))

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination if relevance is very high
            if current_rule_relevance > self.xsi:
                i += 1
                #self.complexity["l.484>xsi"].append((self.sample_to_explain, current_rule_relevance))
                break

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif current_rule_relevance >= best_relevance_so_far:
                best_relevance_so_far = current_rule_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = self._average(sliding_window)
                terminate_threshold = cur_avg_window_relevance / best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition

                print("\n\tCurrent relevance " + str(current_rule_relevance))
                print("\tCurrent averaged window relevance " + str(cur_avg_window_relevance))
                print("\tMax relevance seen so far " + str(best_relevance_so_far))
                print("\tTerminate threshold:" + str(terminate_threshold))
                print("\tRandom value:" + str(random_value))
                print("\tTerminate:" + str(terminate))
                i += 1

        # sort rules by importance 
        all_rules_with_relevance_list = sorted(rule_2_relevance_dict.items(), key=lambda x: x[1], reverse=True)
        self.all_rules_with_relevance = all_rules_with_relevance_list
        self.rule_2_relevance = rule_2_relevance_dict
        return all_rules_with_relevance_list
    

    def extract_rule_relevance_combinatorial(self, length_cap:int) -> list[tuple[tuple,float]]:
        
        best_rule, best_rule_relevance = self.all_rules_with_relevance[0]
        if best_rule_relevance > self.xsi:
            print(f"early exit self.xsi: rules with length 1")
            print(f"best_rule_relevance ({best_rule_relevance}) > self.xsi ({self.xsi})")
            #self.complexity["l.526>xsi"].append((self.sample_to_explain, best_rule_relevance)) 
            
            return self.all_rules_with_relevance, self.complexity
        
        rules_number = len(self.rules_to_remove)
        cur_rule_combinations = 2
        retrainings = 0
        print(f"\n----- Combined Rule Relevance: {2**rules_number-1-rules_number} possible combinations (all lengths considered) -------")

        # stop if you have too few samples (e.g. if you have only 2 samples, you can not extract rules of length 3)
        # or if you get to the length cap
        while cur_rule_combinations <= rules_number and cur_rule_combinations <= length_cap:
            """if self.max_candidates != None:
                if len(self.all_rules_with_relevance) >= max_candidates_here:
                    print(f"number of candidate sets is at {len(self.all_rules_with_relevance)}, max of {max_candidates_here} reached, aborting further search.")
                    self.complexity["max_candidates"] += 1
                    break
            if self.max_retrainings != None:
                if retrainings >= self.max_retrainings:
                    print(f"number of retrainings is at {retrainings}, max of {self.max_retrainings} reached, aborting further search.")
                    self.complexity["max_retrain"] += 1
                    break
            """
            rule_2_relevance = self._extract_rule_combination_relevance(
                length=cur_rule_combinations)
            current_rules_with_relevance = sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)

            self.all_rules_with_relevance += current_rules_with_relevance
            self.all_rules_with_relevance.sort(key=operator.itemgetter(1), reverse=True)
            self.all_rules_with_relevance = self.all_rules_with_relevance[:self.top_k]
            current_best_rule, current_best_rule_relevance = self.all_rules_with_relevance[0]

            if current_best_rule_relevance > best_rule_relevance:
                best_rule, best_rule_relevance = current_best_rule, current_best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit now

            if best_rule_relevance > self.xsi:
                if self.verbose: print(f"early exit self.xsi: cur_rule_length ({cur_rule_combinations}) \
                      \nbest_rule_relevance ({best_rule_relevance}) > self.xsi ({self.xsi})")
                break

            cur_rule_combinations += 1
        combis = copy.deepcopy(self.complexity)
        combis.pop(0)
        combis.pop(1)
        print(f"rule combinations: {combis}")
        
        return self.all_rules_with_relevance[:self.top_k], self.complexity
    
    
    def _extract_rule_combination_relevance(self, length: int):

        
        all_possible_rules_with_preliminary_scores = sorted(
            self.recombinations_w_preliminary_score(length), 
            key=lambda x: x[1], 
            reverse=True)

        rule_2_relevance = {}

        terminate = False
        best_relevance_so_far = ABSURDLY_LOW_VALUE 

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        while i < len(all_possible_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]

            if self.verbose: print(f"\n\tComputing relevance for rule:" + str(current_rule))
            current_rule_relevance = self._compute_relevance_for_rule(flatten_list(current_rule), rule_length=length)  # must flatten list of rules which are combined into one rule (one list of tuples)
            rule_2_relevance[tuple([tuple(rule) for rule in current_rule])] = current_rule_relevance
            if self.verbose: print("\tObtained relevance: " + str(current_rule_relevance))

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = current_rule_relevance

            # early termination
            if current_rule_relevance > self.xsi:
                i += 1
                break

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif current_rule_relevance >= best_relevance_so_far:
                best_relevance_so_far = current_rule_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = self._average(sliding_window)
                terminate_threshold = cur_avg_window_relevance / best_relevance_so_far
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition

                print("\n\tCurrent relevance " + str(current_rule_relevance))
                print("\tCurrent averaged window relevance " + str(cur_avg_window_relevance))
                print("\tMax relevance seen so far " + str(best_relevance_so_far))
                print("\tTerminate threshold:" + str(terminate_threshold))
                print("\tRandom value:" + str(random_value))
                print("\tTerminate:" + str(terminate))
                i += 1

        return rule_2_relevance
    

    def extract_rule_confidence(self, rules:list[tuple[int, int, int]])->dict:
        if self.mode == "necessary":
            confidence = self.ruleset.necessary_pca
        elif self.mode == "sufficient":
            confidence = self.ruleset.sufficient_pca
        
        rule_2_preliminary_relevance = {}

        for rule in rules:

            rule_2_preliminary_relevance[rule] = confidence[rule]
        self.sample_2_relevance = rule_2_preliminary_relevance
        return rule_2_preliminary_relevance
    

    def extract_rule_relevance_incrementally(self, rule_value_list):
        # start with most important rule
        cur_rule_length = 1
        current_set = []
        best_relevance_so_far = ABSURDLY_LOW_VALUE
        total_length = len(rule_value_list)
        retrainings = 0
        while cur_rule_length <= total_length and len(rule_value_list) > 0:
            new_rule, prelim_value = rule_value_list.pop(0)
            current_relevance = self._compute_relevance_for_rule \
                ([*current_set, *new_rule], cur_rule_length)     # unpack list and tuple with *
            retrainings += 1
            if (current_relevance - best_relevance_so_far) > self.improvement_threshhold:
                current_set += new_rule
                best_relevance_so_far = current_relevance
                cur_rule_length += 1
                
                print(best_relevance_so_far, self._pretty_print_list(current_set))
            else:
                print(f"{current_relevance} --- skipped {self._pretty_print_list(new_rule)}")
                continue
            
        if current_set == []:
            raise EmptyResultException(f"no rules explain the prediction '{self.sample_to_explain}'. Rules: {rule_value_list}")
        return [(tuple(current_set), best_relevance_so_far)], self.complexity


    def _compute_relevance_for_rule(self, nple_to_remove:list[tuple[int]], rule_length:int) -> float:
        assert (len(nple_to_remove[0]) == 3), "nple_to_remove must be a list of triples"
        assert self.relevant_triples_set != None, "relevant_triples_set are None, please use setter before calling compute_relevance_for_rule()"

        if rule_length in self.complexity:
            self.complexity[rule_length] += 1
        else:
            self.complexity[rule_length] = 1

        if type(nple_to_remove[0]) == int:
            nple_to_remove = (tuple(nple_to_remove),)   # must be tuple/list of tuple, even if single tuple inside

        # remove duplicate triples from nple_to_remove
        nple_to_remove = list(set(nple_to_remove))


        if self.mode == "necessary":
            relevance, \
            original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
            base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
            pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time = \
                self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                            perspective=self.perspective,
                                            samples_to_remove=nple_to_remove,
                                            relevant_triples=self.relevant_triples_set + (self.sample_to_explain,))

            cur_line = ";".join(self.dataset.sample_to_fact(self.sample_to_explain)) + ";" + \
                    ";".join([";".join(self.dataset.sample_to_fact(x)) for x in nple_to_remove]) + ";" + \
                    str(original_target_entity_score) + ";" + \
                    str(original_target_entity_rank) + ";" + \
                    str(base_pt_target_entity_score) + ";" + \
                    str(base_pt_target_entity_rank) + ";" + \
                    str(pt_target_entity_score) + ";" + \
                    str(pt_target_entity_rank) + ";" + \
                    str(relevance) + ";" + \
                    str(execution_time)

            #end_to_end_string += cur_line + "\n"
            filename_1_details = self.output_prefix + "_10_output_details_" + str(rule_length) + ".csv"
            #print(filename_1_details)
            with open(filename_1_details, "a", encoding="utf8" ) as output_file:
                output_file.writelines([cur_line + "\n"])

            return relevance

        
        if self.mode == "sufficient":

            rule_2_individual_relevances = defaultdict(lambda: [])
            outlines = []

            for j, entity_to_convert in enumerate(self.entities_to_convert):
                print("\t\tConverting entity " + str(j) + " of " + str(len(self.entities_to_convert)) + ": " +
                    self.dataset.entity_id_2_name[entity_to_convert])

                r_nple_to_add = Dataset.replace_entity_in_samples(samples=nple_to_remove,
                                                                old_entity=self.perspective_entity,
                                                                new_entity=entity_to_convert,
                                                                as_numpy=False)
                r_sample_to_convert = Dataset.replace_entity_in_sample(self.sample_to_explain, self.perspective_entity,
                                                                    entity_to_convert)
                r_triple_to_convert = self.dataset.sample_to_fact(r_sample_to_convert)

                # if rule length is 1 try all r_samples_to_add and get their individual relevances
                individual_relevance, \
                original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
                base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
                pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, \
                execution_time = self.engine.addition_relevance(sample_to_convert=r_sample_to_convert,
                                                                perspective=self.perspective,
                                                                samples_to_add=r_nple_to_add)

                rule_2_individual_relevances[tuple(nple_to_remove)].append(individual_relevance)

                outlines.append(";".join(str(self.sample_to_explain)) + ";" + \
                                ";".join(r_triple_to_convert) + ";" + \
                                ";".join([";".join(self.dataset.sample_to_fact(x)) for x in r_nple_to_add]) + ";" + \
                                str(original_best_entity_score) + ";" + \
                                str(original_target_entity_score) + ";" + \
                                str(original_target_entity_rank) + ";" + \
                                str(base_pt_best_entity_score) + ";" + \
                                str(base_pt_target_entity_score) + ";" + \
                                str(base_pt_target_entity_rank) + ";" + \
                                str(pt_best_entity_score) + ";" + \
                                str(pt_target_entity_score) + ";" + \
                                str(pt_target_entity_rank) + ";" + \
                                str(execution_time) + ";" + \
                                str(individual_relevance))

            # add the rule global relevance to all the outlines that refer to this rule
            global_relevance = self._average(rule_2_individual_relevances[tuple(nple_to_remove)])

            complete_outlines = [x + ";" + str(global_relevance) + "\n" for x in outlines]

            filename1 = self.output_prefix + "_10_output_details_" + str(rule_length) + ".csv"
            print(filename1)
            with open(filename1, "a", encoding="utf8" ) as output_file:
                output_file.writelines(complete_outlines)

            return global_relevance
    
    
    def short_print_rule_relevance(self, rule_samples_with_relevance: dict|list, inplace=True):
        if type(rule_samples_with_relevance) == dict:
            rule_samples_with_relevance = sorted(rule_samples_with_relevance.items(), key=lambda x: x[1], reverse=True)
        result = ""
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance
            if inplace:
                print(f"{round(float(cur_relevance), 3)} : {str(self.ruleset.extract_relations(cur_rule_samples, self.dataset))}")
            else:
                result += f"{round(float(cur_relevance), 3)} : {str(self.ruleset.extract_relations(cur_rule_samples, self.dataset))}; "
        return result


    def recombinations_w_preliminary_score(self, length):
        all_possible_rule_combinations = itertools.combinations(self.rules_to_remove, length)
        all_possible_rules_with_preliminary_scores = [(x, self._calc_preliminary_rule_score(tuple(flatten_list(x)))) 
                                                      for x in all_possible_rule_combinations]
        return all_possible_rules_with_preliminary_scores

    
    def _calc_preliminary_rule_score(self, rule: tuple[tuple[int, int, int]]) -> float:
        try:
            result = (numpy.sum([self.sample_2_relevance[x] for x in rule])/len(rule))   #avg importance
        except TypeError as e:
            if "list" in str(e):
                print(f"TypeError: list not hashable, must be tuple!")
            raise TypeError
        return result
    

    @staticmethod
    def _average(l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))


    def _pretty_print_list(self, liste:list)->str:
        print_list = ""
        if type(liste[0])==int:
            liste = [liste]

        for item in liste:
            print_list += (f"{str(self.dataset.sample_to_fact(item))}")
        return str(print_list)
    
    def _pretty_print_dict(self, dictionary:dict)->str:
        print_dictionary = ""
        for key, value in dictionary.items():
            print_dictionary += (f"{value} : {str(self.dataset.sample_to_fact(key))}\n")
        return str(print_dictionary)
    
class EmptyResultException(Exception):
    pass

class RuleNotApplicableException(Exception):
    pass
