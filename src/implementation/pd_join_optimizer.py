import pandas as pd
import random
import networkx as nx
import math
from enum import IntEnum
import logging
from functools import reduce
from typing import Generator, Optional, Tuple, Union
from abc import ABC
from types import GeneratorType

# Control whether index should be set prior to merging DataFrames.
SET_INDEX = False

class Strategy(ABC):
    pass

class BaselineStrategy(Strategy):
    def __init__(self, join_order: list[list[int]]):
        super().__init__()
        assert join_order is not None, 'Join order cannot be None'
        self.join_order = join_order

class AcyclicStrategy(Strategy):
    def __init__(self, tree: nx.Graph, root: int):
        super().__init__()
        assert tree is not None, 'tree cannot be None'
        assert root is not None and root in tree.nodes, 'root must be a node in the tree'
        self.tree = tree
        self.root = root
        self.postorder: Optional[list[tuple[int, int]]] = None
        self.preorder: Optional[list[tuple[int, int]]] = None
    
    @classmethod
    def precomputed(cls, tree: nx.Graph, root: int, preorder: list[tuple[int, int]], postorder: list[tuple[int, int]]):
        strategy = cls(tree, root)
        assert preorder is not None, 'preorder cannot be None'
        assert postorder is not None, 'postorder cannot be None'
        strategy.preorder = preorder
        strategy.postorder = postorder
        return strategy
    
    def has_precomputed_traversals(self):
        return self.preorder is not None and self.postorder is not None

class VariableSelectionStrategy(IntEnum):
    RANDOM = 1
    LOWEST_DISTINCT_COUNT = 2
    
# A partitioning strategy for a join involving three relations includes:
# 1. the partitioning variable selection strategy
# 2. optionally:
#    * the indices of the relations R, S, T, whereas R is the relation to be partitioned
#    * the variable of relation R to be used for partitioning
class PartitioningStrategy(Strategy):
    def __init__(self, variable_selection_strategy: VariableSelectionStrategy, reduce_duplicates: bool):
        assert variable_selection_strategy is not None, "Variable selection strategy must not be None"
        self.variable_selection_strategy = variable_selection_strategy
        self.reduce_duplicates = reduce_duplicates
    
    @classmethod
    def precomputed(cls, variable_selection_strategy: VariableSelectionStrategy, reduce_duplicates: bool, indices: dict[str, int], variable: str):
        obj = cls(variable_selection_strategy, reduce_duplicates)

        assert indices is not None and {0, 1, 2} == set(indices.values()), "Indices may only contain the values 0, 1, and 2"
        assert 'R' in indices and 'S' in indices and 'T' in indices, "Indices must contain the keys 'R', 'S', and 'T'"
        assert variable is not None and len(variable) > 0, "Variable must not be None and must not be empty"

        obj.variable = variable
        obj.indices = indices

        return obj

    def has_precomputed_values(self):
        return hasattr(self, 'indices') and hasattr(self, 'variable')

# Given a mapping of key to list of values and a key, determine the unique values associated with the given key, i.e. that are those that do not occur in a list with a different key.
# Return a list of unique values.
def get_unique_values(key_to_values: dict[int, list[str]], key: int) -> set[str]:
    # Get a copy of key_to_values
    unique_values = set()

    # Check if there is a value that does not exist in any other list of values with a different key
    for value in key_to_values[key]:
        unique = True

        for other_key, other_values in key_to_values.items():
            if key != other_key and value in other_values:
                unique = False
                break

        # If there exists such a value, add it to the list of unique values
        if unique:
            unique_values.add(value)
    
    return unique_values

# Given a mapping of keys to lists of values, and a pair (key, values).
# Check if there exists a list with a different key that contains all values.
# Return the key of the list if it exists, None otherwise.
def get_superset_key(index_to_column: dict[int, list[str]], key: int, values: list[str]) -> Optional[int]:
    # Add non-determinism to the order in which entries are processed
    items = list(index_to_column.items())
    random.shuffle(items)

    for other_key, other_values in items:
        if key != other_key and set(values).issubset(other_values):
            return other_key
    
    return None

# Do the following until no more changes are made to variable index_to_column during an iteration:
# 1. Remove an entry from index_to_column, if its values are unique. 
# 2. Remove an entry from index_to_column, if its non-unique values are a subset of another entry's values.
def perform_gyo_prime_reduction(index_to_columns: dict[int, list[str]]) -> Union[tuple[list[int], dict[int, int]], tuple[None, None]]:
    index_to_columns = index_to_columns.copy()
    logging.debug('GYO reduction: {}'.format(index_to_columns))

    key_deletion_sequence = []
    superset_key_of_deleted_keys = {}

    while True:
        # Get a copy of index_to_column
        index_to_column_copy = {key: values.copy() for key, values in index_to_columns.items()}

        # Add non-determinism to the order in which entries are processed
        items = list(index_to_column_copy.items())
        random.shuffle(items)

        for key, values in items:
            # Get unique values
            unique_values = get_unique_values(index_to_columns, key)
            
            # Get values that exist in lists with different keys
            non_unique_values = [value for value in values if value not in unique_values]

            if len(non_unique_values) == 0:
                # The atom shares no variable with other atoms
                key_deletion_sequence.append(key)
                del index_to_columns[key]
                break

            # Check if there exists a list with a different key that contains all non-unique values
            superset_key = get_superset_key(index_to_columns, key, non_unique_values)

            if superset_key != None:
                # There exists a witness R' s.t. each variable in R either appears in R only, or also appears in R'
                superset_key_of_deleted_keys[key] = superset_key
                key_deletion_sequence.append(key)
                del index_to_columns[key]
                break            
        
        logging.debug('GYO reduction: {}'.format(index_to_columns))

        # Check if any changes were made
        if index_to_columns == index_to_column_copy:
            break
    
    logging.debug('key_deletion_sequence: {}'.format(key_deletion_sequence))
    logging.debug('superset_key_of_deleted_keys: {}'.format(superset_key_of_deleted_keys))

    if len(index_to_columns) == 0:
        return (key_deletion_sequence, superset_key_of_deleted_keys)
    else:
        return (None, None)
    
# Given a nx.Graph, we determine the root node such that the depth of the tree rooted at that node is minimal.
# The depth of a tree rooted at some node is the maximum path length from that node to any leaf node.
def choose_root_with_minimal_depth(graph: nx.Graph) -> Optional[int]:
    # If the graph is not connected, we cannot determine a root node.
    if not nx.is_connected(graph):
        return None
    
    min_max_depth = float('inf')
    best_root = None

    for node in graph.nodes:
        depths = nx.shortest_path_length(graph, source=node)
        max_depth = max(depths.values())
        if max_depth < min_max_depth:
            min_max_depth = max_depth
            best_root = node

    return best_root

def construct_join_tree(key_deletion_sequence: list[int], superset_key_of_deleted_keys: dict[int, int]) -> Optional[Tuple[nx.Graph, int]]: 
    if key_deletion_sequence == None or superset_key_of_deleted_keys == None:
        return None
    
    logging.info("Constructing join tree")
    logging.debug("key_deletion_sequence: {} and superset_key_of_deleted_keys: {}".format(key_deletion_sequence, superset_key_of_deleted_keys))

    G = nx.Graph()
    prior_node = None

    for key, value in superset_key_of_deleted_keys.items():
        G.add_edge(key, value)

    # Get keys from key_deletion_sequence that are neither keys nor values in superset_key_of_deleted_keys
    not_added_keys = [value for value in key_deletion_sequence if value not in superset_key_of_deleted_keys.keys() and value not in superset_key_of_deleted_keys.values()]

    # Add remaining keys to the graph
    for value in not_added_keys:
        G.add_node(value)

    # Check if the graph is a forest
    # If yes, we connect all trees such that the resulting graph is a tree
    if nx.is_forest(G):
        # Get all trees
        trees = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        # Connect all trees
        for i in range(0, len(trees)):
            # Get some node of the current tree
            candidates = list(trees[i].nodes)
            current_node = random.sample(candidates, len(candidates))[0]

            if i > 0:
                # Add an edge between the node of the current tree and the node from the previous tree
                trees[0].add_edge(prior_node, current_node)

            # Set the node of the prior tree to the node of the current tree
            prior_node = current_node

        # Set the graph to the first tree to which all other trees have been connected
        G = trees[0]
    
    # Choose root node in G such that the depth of the tree rooted at that node is minimal
    root = choose_root_with_minimal_depth(G)
    
    logging.debug("Root {}".format(root))

    return (G, root)

def apply_yannakakis_algorithm(strategy: AcyclicStrategy, dataframes: list[pd.DataFrame]) -> Tuple[pd.DataFrame, AcyclicStrategy]:
    logging.info("Applying Yannakakis algorithm")
    graph, root = strategy.tree, strategy.root

    if strategy.has_precomputed_traversals():
        logging.debug('Using precomputed postorder and preorder traversals')
        postorder = strategy.postorder
    else:
        postorder = get_reordered_postorder(dfs_postorder_nodes(graph, root), dataframes)
        strategy.postorder = []
        strategy.preorder = dfs_preorder_nodes(graph, root)

    # Perform 1st bottom up traversal of graph: compute semi-joins (upwards propagation)
    logging.debug('Performing 1st bottom up traversal of graph: compute semi-joins (upwards propagation)')

    for (vertex, parent) in postorder:
        # Perform semi-join
        logging.debug(f'Performing left semi-join on ({parent}, {vertex})')
        on = list(get_set_of_shared_variables(dataframes[parent], dataframes[vertex]))
        dataframes[parent] = perform_left_semi_join(dataframes[parent], dataframes[vertex], on)

        if isinstance(postorder, GeneratorType):
            strategy.postorder.append((vertex, parent))

    # If the resulting dataframe at the root node is empty, return empty DataFrame
    if dataframes[root].empty:
        logging.debug('The resulting dataframe at the root node is empty, returning empty DataFrame')
        dataframes[root] = pd.DataFrame(columns=list(reduce(lambda x, y: x.union(y), [set(df.columns) for df in dataframes])))
    else:
        # Perform top-down traversal: "reverse" semi-joins (downwards propagation)
        logging.debug('Performing top-down traversal: "reverse" semi-joins (downwards propagation)')

        for (vertex, parent) in strategy.preorder:
            # Perform "reverse" semijoin
            logging.debug(f'Performing left semi-join on ({vertex}, {parent})')
            on = list(get_set_of_shared_variables(dataframes[vertex], dataframes[parent]))
            dataframes[vertex] = perform_left_semi_join(dataframes[vertex], dataframes[parent], on)

        # Perform 2nd bottom up traversal of graph: compute solution using joins
        logging.debug('Performing 2nd bottom up traversal of graph: compute solution using joins')

        for (vertex, parent) in strategy.postorder:
            # Merge the dataframes associated with vertex and parent indices
            logging.debug(f'Merging dataframes associated with {vertex} and {parent}')
            dataframes[parent] = perform_natural_join(dataframes[parent], dataframes[vertex])

            # After the join, we can remove the dataframe associated with vertex
            vertex_to_remove = dataframes[vertex]
            dataframes[vertex] = None
            del vertex_to_remove
    
    # Update strategy
    strategy = AcyclicStrategy.precomputed(graph, root, strategy.preorder, strategy.postorder)

    return dataframes[root], strategy

def dfs_postorder_nodes(graph: nx.Graph, root: int) -> list[tuple[int, int]]:
    # Computes the inverse depth for each node in the graph and a mapping of each node to its parent.
    # The inverse depth for a node is the maximum inverse depth of its children plus one.
    def compute_inverse_depth_and_parent_maps(graph, node, visited):
        visited.add(node)
        depth = 0

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                parent_map[neighbor] = node
                compute_inverse_depth_and_parent_maps(graph, neighbor, visited)
                depth = max(depth, depth_map[neighbor])

        depth_map[node] = depth + 1

    # Using the inverse depth and parent maps, we compute the post-order sequence of nodes.
    # We visit the children of a node in descending order of their inverse depth.
    # As a result, leave nodes with higher depth are visited first.
    def compute_post_order_sequence(graph, node, visited):
        visited.add(node)

        for neighbor in sorted(graph.neighbors(node), key=lambda x: depth_map[x], reverse=True):
            if neighbor not in visited:
                compute_post_order_sequence(graph, neighbor, visited)
        
        parent = None if node == root else parent_map[node]

        if parent is not None:
            sequence.append((node, parent))
        
    depth_map = {}
    parent_map = {}
    sequence = []
    
    compute_inverse_depth_and_parent_maps(graph, root, visited=set())
    compute_post_order_sequence(graph, root, visited=set())

    return sequence

# Given a postorder traversal of a tree with nodes being associated to DataFrame objects, i.e. a list of pairs [(u_1, v_1), (u_2, v_2), ..., (u_n, v_n)] where v_i is the parent of u_i for 1 <= i <= n.
# Return the first consecutive sublist of pairs [(u_1, parent), (u_2, parent), ..., (u_n, parent)] sharing the same parent, such that the following holds:
# If the size of the DataFrame associated with u_i is smaller than the size of the DataFrame associated with u_j, then the pair (u_i, parent) comes before the pair (u_j, parent).
# The idea behind that is that we want to perform the left semi-join first that has the smallest result in the right argument, because we assume that this creates the smallest intermediate result and thus the remaining semi-joins will be faster for a specific parent node.
# The order of the parents may not be changed, i.e. the pairs (u_i, v_i) and (u_j, v_j) may not be swapped if v_i != v_j.
def get_first_reordered_postorder_sublist(postorder: list[tuple[int, int]], dataframes: list[pd.DataFrame]) -> list[tuple[int, int]]:
    sublist = []

    for (vertex, parent) in postorder:
        # If the parent of the current vertex is different to a parent of the sublist, then we can stop as we have found the first consecutive sublist of vertices sharing the same parent
        if len(sublist) != 0 and sublist[0][1] != parent:
            break
        
        # Append the pair (vertex, parent) to the sublist
        sublist.append((vertex, parent))
    
    # Sort the sublist such that the vertex associated with the DataFrame having the smallest number of rows comes first
    sublist.sort(key=lambda x: dataframes[x[0]].shape[0])

    return sublist

# Given a postorder traversal of a tree with nodes being associated to DataFrame objects, i.e. a list of pairs [(u_1, v_1), (u_2, v_2), ..., (u_n, v_n)] where v_i is the parent of u_i for 1 <= i <= n.
# The generator yields a single pair (u_i, v_i) by repeatedly calling "get_first_reordered_postorder_sublist" with different starting indices for the first argument until the returned sublist is empty.
# For example, if the first sublist contains a list of length x < n, then the second call should include all elements after the first x elements of the postorder traversal as first argument.
def get_reordered_postorder(postorder: list[tuple[int, int]], dataframes: list[pd.DataFrame]) -> Generator[tuple[int, int], None, None]:
    i = 0

    while i < len(postorder):
        sublist = get_first_reordered_postorder_sublist(postorder[i:], dataframes)

        if len(sublist) == 0:
            break
        
        for pair in sublist:
            yield pair
        
        i += len(sublist)

# Based on https://networkx.org/documentation/latest/_modules/networkx/algorithms/traversal/depth_first_search.html#dfs_preorder_nodes
# Added parent vertex as second element of each pair returned
def dfs_preorder_nodes(graph: nx.Graph, source: int) -> list[tuple[int, int]]:
    edges = nx.dfs_labeled_edges(graph, source=source)
    return list(((v, u) for u, v, d in edges if d == "forward" and u != v))

# Determine the set of shared variables between two dataframes
def get_set_of_shared_variables(dataframe_x: pd.DataFrame, dataframe_y: pd.DataFrame) -> set[str]:
    return set(dataframe_x.columns.values.tolist()).intersection(set(dataframe_y.columns.values.tolist()))

def perform_regular_join(df1: pd.DataFrame, df2: pd.DataFrame, on) -> pd.DataFrame:
    # If there are no shared variables, then we perform a cross join.
    if len(on) == 0:
        logging.debug('Performing cross join')
        return pd.merge(df1, df2, on=None, how='cross', copy=False)
    else:        
        # Otherwise, we perform an inner join.
        logging.debug('Sizes of dataframes: {} {}'.format(len(df1), len(df2)))
        if SET_INDEX:
            result = pd.merge(df1.set_index(on), df2.set_index(on), how='inner', left_index=True, right_index=True, copy=False, sort=False).reset_index()
        else:
            result = pd.merge(df1, df2, how='inner', on=on, copy=False)

        return result

def perform_left_semi_join(left_df: pd.DataFrame, right_df: pd.DataFrame, on) -> pd.DataFrame:
    # Based on code from closed GitHub pull request, which aimed to add support for a left semi-join to pandas.
    # https://github.com/pandas-dev/pandas/pull/49661
    # Important notice: This does not perform datatype validations on the colums defined in the "on" argument between the two dataframes that are normally performed by pandas as part of the merge() function.
    def _semi_helper(
        leftdf: pd.DataFrame,
        left: Union[pd.Index, pd.DataFrame],
        right: Union[pd.Index, pd.DataFrame],
    ) -> pd.DataFrame:
        if not isinstance(left, pd.Index):
            if len(left.columns) == 1:
                left = pd.Index(left.values.flatten())
            else:
                left = pd.MultiIndex.from_frame(left)
        if not isinstance(right, pd.Index):
            if len(right.columns) == 1:
                right = pd.Index(right.values.flatten())
            else:
                right = pd.MultiIndex.from_frame(right)
        subset = left.isin(right)
        return leftdf.loc[subset]
    
    # If there are no shared variables, then the join is a cross join
    # In this case, no semi-join is performed
    if len(on) != 0:
        logging.debug('Sizes of dataframes: {} {}'.format(len(left_df), len(right_df)))
        left_df = _semi_helper(left_df, left_df[on], right_df[on])
        logging.debug('Size of resulting dataframe: {}'.format(len(left_df)))
    
    return left_df

def perform_natural_join(left_df: pd.DataFrame, right_df: pd.DataFrame, reduce_duplicates: bool = False) -> pd.DataFrame:
    on = list(get_set_of_shared_variables(left_df, right_df))

    # If `right_df` does have a variable that does not occur in `left_df`, then we perform a regular join
    # Otherwise, we perform a semi-join
    if len(set(right_df.columns.values.tolist()) - set(left_df.columns.values.tolist())) > 0:
        return perform_regular_join(left_df, right_df, on)
    else:
        if reduce_duplicates:
            # If we are not interested in duplicates, then we can reduce the number of duplicates by performing a left semi join
            # This would not produce the correct amount of tuples – compared to a regular join – in case there is more than one matching tuple in right_df
            return perform_left_semi_join(left_df, right_df, on)
        else:
            return perform_regular_join(left_df, right_df, on)
    
def apply_partitioning_based_join_algorithm(dataframes: list[pd.DataFrame],
                                            strategy: PartitioningStrategy) -> Tuple[pd.DataFrame, Strategy]:
    # In case of a join involving more than three relations, we do not perform any optimization.
    # Instead, the join will be performed as a sequence of two-way joins.
    if len(dataframes) != 3:
        return join_baseline(dataframes)

    logging.info("Applying partitioning-based join algorithm")

    if strategy.has_precomputed_values():
        logging.debug("Using precomputed values for partitioning")
        R_index, S_index, T_index = strategy.indices['R'], strategy.indices['S'], strategy.indices['T']
        R, S, T = dataframes[R_index], dataframes[S_index], dataframes[T_index]

        # Determine the set of shared variables between R and S, and between R and T
        shared_variables_R_S, shared_variables_R_T = get_set_of_shared_variables(R, S), get_set_of_shared_variables(R, T)

        # Get the shared variables that either occur in R and S, or in R and T, but not both.
        shared_variables = shared_variables_R_S.union(shared_variables_R_T) - shared_variables_R_S.intersection(shared_variables_R_T)
        
        if strategy.variable not in shared_variables:
            raise ValueError(f'It does not hold for the user-specified partitioning variable {strategy.variable} that it either occurs in R and S, or in R and T, but not both.')
    else:
        excluded_indices = []
        shared_variables = set()

        while len(excluded_indices) != len(dataframes) and shared_variables == set():
            # Get the indices of the DataFrames that are not excluded
            indices = [i for i in range(len(dataframes)) if i not in excluded_indices]

            # Choose R to be the relation with the smallest number of tuples
            R_index = min(range(len(indices)), key=lambda i: dataframes[indices[i]].shape[0])
            R = dataframes[R_index]

            # Let S and T be the other two relations
            S_index, T_index = 0 if R_index != 0 else 1, 2 if R_index != 2 else 1
            S, T = dataframes[S_index], dataframes[T_index]

            # Determine the set of shared variables between R and S, and between R and T
            shared_variables_R_S, shared_variables_R_T = get_set_of_shared_variables(R, S), get_set_of_shared_variables(R, T)

            # Get the shared variables that either occur in R and S, or in R and T, but not both.
            shared_variables = shared_variables_R_S.union(shared_variables_R_T) - shared_variables_R_S.intersection(shared_variables_R_T)

            if shared_variables == set():
                # If there are no variables that either occur in R and S, or in R and T, but not both, we choose a different R for partitioning
                excluded_indices.append(R_index)

        if shared_variables == set():
            # If no relation could be chosen to be partitioned, we fall back to a sequence of two-way joins
            logging.debug('No relation could be chosen to be partitioned. Falling back to a sequence of two-way joins.')
            return join_baseline(dataframes)
    
    logging.debug(f'R -> {R_index}: {R.columns.to_list()}, S -> {S_index}: {S.columns.to_list()}, T -> {T_index}: {T.columns.to_list()}')
    
    distinct_values = {}

    if strategy.has_precomputed_values():
        # Use the user-specified partitioning variable
        partitioning_variable = strategy.variable
        distinct_values[partitioning_variable] = R[partitioning_variable].value_counts()
    elif strategy.variable_selection_strategy == VariableSelectionStrategy.RANDOM:
        # Choose a random variable as the partitioning variable
        logging.debug('Choosing the partitioning variable from R randomly')
        partitioning_variable = random.choice(list(shared_variables))
        distinct_values[partitioning_variable] = R[partitioning_variable].value_counts()
    elif strategy.variable_selection_strategy == VariableSelectionStrategy.LOWEST_DISTINCT_COUNT:
        # For each shared variable, determine number of distinct values within R. 
        # We choose the variable with the lowest number of distinct values as the partitioning variable.
        logging.debug('Choosing the variable of R with the lowest number of distinct values as the partitioning variable')
        min_distinct_values = -1
        partitioning_variable = None

        for column in list(shared_variables):
            distinct_values[column] = R[column].value_counts()
            number_of_distinct_values = distinct_values[column].shape[0]

            if min_distinct_values == -1 or number_of_distinct_values < min_distinct_values:
                min_distinct_values = number_of_distinct_values
                partitioning_variable = column
    
    logging.debug(f'R is partitioned based on variable {partitioning_variable}')

    # Determine the frequency of each value within variable partitioning_variable in R
    # Avoid scanning the column again if we already did so before
    frequency = distinct_values[partitioning_variable] if partitioning_variable in distinct_values else R[partitioning_variable].value_counts()

    # Set partitioning threshold
    partitioning_threshold = math.sqrt(R.shape[0])

    # Split R into two partitions
    R_heavy = R[R[partitioning_variable].isin(frequency[frequency > partitioning_threshold].index)]
    R_light = R[R[partitioning_variable].isin(frequency[frequency <= partitioning_threshold].index)]

    logging.debug(f'R+: {R_heavy.shape[0]} tuples, R-: {R_light.shape[0]} tuples, S: {S.shape[0]} tuples, T: {T.shape[0]} tuples')

    # Join R_heavy first with the dataframe that does not contain the partitioning variable
    # Join R_light first with the dataframe that does contain the partitioning variable
    # Then join each partition with the remaining dataframe
    log_suffix = ' (reducing duplicates)' if strategy.reduce_duplicates else ''
                  
    if partitioning_variable in shared_variables_R_S:
        # Join R_heavy with T first
        logging.debug('Computing R+ ⋈ T')
        heavy_result = perform_regular_join(R_heavy, T, on=list(shared_variables_R_T))

        logging.debug('Computing (R+ ⋈ T) ⋈ S' + log_suffix)
        heavy_result = perform_natural_join(left_df=heavy_result, right_df=S, reduce_duplicates=strategy.reduce_duplicates)

        # Join R_light with S first
        logging.debug('Computing R- ⋈ S')
        light_result = perform_regular_join(R_light, S, on=list(shared_variables_R_S))

        logging.debug('Computing (R- ⋈ S) ⋈ T' + log_suffix)
        light_result = perform_natural_join(left_df=light_result, right_df=T, reduce_duplicates=strategy.reduce_duplicates)
    else:
        # Join R_heavy with S first
        logging.debug('Computing R+ ⋈ S')
        heavy_result = perform_regular_join(R_heavy, S, on=list(shared_variables_R_S))

        logging.debug('Computing (R+ ⋈ S) ⋈ T' + log_suffix)
        heavy_result = perform_natural_join(left_df=heavy_result, right_df=T, reduce_duplicates=strategy.reduce_duplicates)

        # Join R_light with T first
        logging.debug('Computing R- ⋈ T')
        light_result = perform_regular_join(R_light, T, on=list(shared_variables_R_T))

        logging.debug('Computing (R- ⋈ T) ⋈ S' + log_suffix)
        light_result = perform_natural_join(left_df=light_result, right_df=S, reduce_duplicates=strategy.reduce_duplicates)
        
    # Combine heavy and light results
    result = pd.concat([heavy_result, light_result], ignore_index=True, copy=False)

    # Update strategy
    strategy = PartitioningStrategy.precomputed(variable_selection_strategy=strategy.variable_selection_strategy, 
                                                reduce_duplicates=strategy.reduce_duplicates, 
                                                variable=partitioning_variable, 
                                                indices={'R': R_index, 'S': S_index, 'T': T_index})
    
    return result, strategy

def perform_join_as_sequence_of_two_way_joins(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    join_result = dataframes[0]

    for i in range(1, len(dataframes)):
        join_result = perform_natural_join(join_result, dataframes[i])
    
    return join_result

# Given is a list of DataFrame objects to be joined, with each DataFrame representing a relation.
# First, a graph G = (V, E) is created where V represents the relations and E consists of all pairs of relations that share at least one variable.
# Each edge indicates that an inner join can be performed between the two relations.
# This graph can consist of multiple connected components, in case a cross join is required.
# But in practice, the graph will consist of a single connected component in most cases.
# Within each connected component, a traversal is performed to determine the order in which the relations should be joined.
# After a join order has been determined, the joins are performed in the order determined by the traversal.
# As a last step, the results of the connected components are combined with a cross join.
# A user may optionally specify a join order within the "strategy" argument to override the join order determined by the traversal.
# The join order consists of a list of lists, where each sublist represents the join order for a connected component.
# This parameter is useful for repeating a query execution with the same join order during evaluation.
def join_baseline(dataframes: list[pd.DataFrame], strategy: BaselineStrategy = None) -> Tuple[pd.DataFrame, BaselineStrategy]:
    logging.info('Performing enhanced baseline join')

    # Output column names for debugging purposes
    for i in range(len(dataframes)):
        logging.debug(f'Relation {i}: {list(dataframes[i].columns)}')

    # Construct graph G = (V, E)
    graph = construct_graph(dataframes)

    # Compute join order for each connected component
    computed_join_order = list(map(lambda component: determine_join_order(graph, component), nx.connected_components(graph)))

    logging.debug(f'Computed join order: {computed_join_order}')

    # Override join order if user has specified a join order
    if strategy is not None and strategy.join_order is not None:
        logging.debug(f'Overriding join order with user-specified join order: {strategy.join_order}')

        # The number of connected components must match.
        if len(strategy.join_order) != len(computed_join_order):
            raise ValueError('The number of connected components in the user-specified join order does not match the number of connected components in the computed join order.')
        
        # Check if there are duplicate relations in any connected component of the user-specified join order
        for component_join_order in strategy.join_order:
            if len(component_join_order) != len(set(component_join_order)):
                raise ValueError(f'The user-specified join order contains duplicate relations in a connected component: {component_join_order}')
            
        # Make a copy of both the precomputed join order and the user-specified join order such that we have a list of sets
        computed_join_order_sets = list(map(lambda component_join_order: set(component_join_order), computed_join_order))
        join_order_sets = list(map(lambda component_join_order: set(component_join_order), strategy.join_order))

        for component_join_order_set in computed_join_order_sets:
            # Check if the connected component of the precomputed join order is contained in the user-specified join order
            if component_join_order_set not in join_order_sets:
                raise ValueError(f'The user-specified join order does not contain the same relations as the computed join order in one connected component: {component_join_order_set}')
    else:
        strategy = BaselineStrategy(computed_join_order)
    
    # First, join each sublist of relations associated with a connected component in the given join order
    intermediate_results = [perform_join_as_sequence_of_two_way_joins([dataframes[i] for i in component_join_order]) for component_join_order in strategy.join_order]
    
    # Then, join the intermediate results of the connected components with a cross join
    return perform_join_as_sequence_of_two_way_joins(intermediate_results), strategy

# Given is a list of DataFrame objects, with each DataFrame representing a relation.
# The function returns a graph G = (V, E), where V represents the relations and E consists of all pairs of relations that share at least one variable.
def construct_graph(dataframes: list[pd.DataFrame]) -> nx.Graph:
    graph = nx.Graph()

    # Add each relation – i.e. the index within the list of dataframes – as vertex to the graph 
    for i in range(len(dataframes)):
        graph.add_node(i)

    # Check each pair of relations for common variables
    # If there is at least one common variable between two relations, an edge between the two relations is added to the graph
    for i in range(len(dataframes)):
        for j in range(i + 1, len(dataframes)):
            if len(set(dataframes[i].columns).intersection(dataframes[j].columns)) > 0:
                graph.add_edge(i, j)

    return graph

# Given is a graph G = (V, E) and a connected component of G.
# A depth-first traversal is performed in the connected component.
# Both the starting point of a traversal and the vertex to be visited next are chosen randomly.
# The function returns a list of integers – i.e. the indices of the relations – that specifies the order in which the relations should be joined.
def determine_join_order(graph: nx.Graph, component: set[int]) -> list[int]:
    # Choose random vertex within the connected component as starting point
    start = random.choice(list(component))

    # Perform depth-first traversal
    join_order = []
    visited = set()

    def dfs(v):
        visited.add(v)
        join_order.append(v)

        # Choose random unvisited neighbor as next vertex to be visited
        neighbors = list(graph.neighbors(v))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)

    return join_order

# This join method applies optimizations known from database theory.
# It utilizes the GYO reduction to determine if the given join is acyclic or not.
# In case it is acyclic, a join tree is constructed and the join is computed using the Yannakakis algorithm.
# In case it is cyclic, we perform a worst-case optimal partitioning-based join approach if the number of given DataFrames is equal to 3.
# Otherwise, we utilize the baseline join method.
def join_optimized(dataframes: list[pd.DataFrame], 
                   strategy: Strategy = None, 
                   reduce_duplicates: bool = False,
                   variable_selection_strategy: VariableSelectionStrategy = VariableSelectionStrategy.LOWEST_DISTINCT_COUNT) -> Tuple[pd.DataFrame, Strategy]:
    # Output column names for debugging purposes
    for i in range(len(dataframes)):
        logging.info(f'Relation {i}: {list(dataframes[i].columns)}')

    result = perform_gyo_prime_reduction(index_to_columns={i: df.columns.values.tolist() for i, df in enumerate(dataframes)})

    if result != (None, None):
        logging.info('Join is acyclic')
        (key_deletion_sequence, superset_key_of_deleted_keys) = result

        # Construct join tree
        (G, root) = construct_join_tree(key_deletion_sequence, superset_key_of_deleted_keys)

        if strategy is not None:
            assert type(strategy) == AcyclicStrategy, 'The given strategy is not applicable for acyclic joins'
        else:
            strategy = AcyclicStrategy(G, root)

        # Apply Yannakakis algorithm
        return apply_yannakakis_algorithm(strategy, dataframes)
    else:
        logging.info('Join is cyclic')

        if strategy is None:
            strategy = PartitioningStrategy(variable_selection_strategy=variable_selection_strategy, reduce_duplicates=reduce_duplicates)
        else:
            if type(strategy) == BaselineStrategy:
                return join_baseline(dataframes, strategy)
            elif type(strategy) == AcyclicStrategy:
                raise ValueError('The given acyclic strategy is not applicable for cyclic joins')

        # Apply partitioning-based join algorithm for cyclic join
        # In case of a cyclic join involving more than three relations, the algorithm will not perform any optimization, i.e. it will call the baseline join method.
        # If no relation can be found to be partitioned with a join variable that exists in exactly one additional relation, the algorithm will fall back to the baseline join method as well.
        return apply_partitioning_based_join_algorithm(dataframes, strategy)