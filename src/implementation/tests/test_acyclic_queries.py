import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pd_join_optimizer import join_optimized, construct_join_tree, perform_gyo_prime_reduction
import networkx as nx

def get_dataframes_of_acyclic_query_from_database_theory_lecture():
    r_1 = pd.DataFrame({'x_3': [('b_1'), ('b_2'), ('b_3')]})
    r_2 = pd.DataFrame({'x_2': [('c_1'), ('c_1'), ('c_1')], 'x_4': [('a_1'), ('a_1'), ('a_2')], 'x_3': [('b_1'), ('b_2'), ('b_2')]})
    r_3 = pd.DataFrame({'x_1': [('s_1'), ('s_1'), ('s_3'), ('s_3'), ('s_2')], 'x_2': [('c_1'), ('c_1'), ('c_3'), ('c_1'), ('c_2')], 'x_3': [('b_1'), ('b_2'), ('b_1'), ('b_4'), ('b_3')]})
    r_4 = pd.DataFrame({'x_2': [('c_1'), ('c_1'), ('c_4')], 'x_3': [('b_2'), ('b_1'), ('b_6')]})
    r_5 = pd.DataFrame({'x_5': [('c_1'), ('c_1'), ('c_4')], 'x_6': [('b_2'), ('b_1'), ('b_6')]})
    
    return [r_1, r_2, r_3, r_4, r_5]

def test_join_acyclic_query():
    dataframes = get_dataframes_of_acyclic_query_from_database_theory_lecture()
    result, _ = join_optimized(dataframes)

    # Order columns and rows of result
    result = result[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']]
    result = result.sort_values(by=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']).reset_index(drop=True)

    # Get ground truth
    dataframes = get_dataframes_of_acyclic_query_from_database_theory_lecture()
    ground_truth = pd.merge(pd.merge(pd.merge(pd.merge(dataframes[0], dataframes[1]), dataframes[2]), dataframes[3]), dataframes[4], how='cross')
    ground_truth = ground_truth[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']]
    ground_truth = ground_truth.sort_values(by=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']).reset_index(drop=True)

    # Assert that the result is equal to the expected result
    assert result.equals(ground_truth), 'The result of performing the join of the "Database Theory" lecture example is not equal to the expected result'

def test_join_tree_construction():
    dataframes = get_dataframes_of_acyclic_query_from_database_theory_lecture()
    index_to_columns = {i: df.columns.values.tolist() for i, df in enumerate(dataframes)}
    (key_deletion_sequence, superset_key_of_deleted_keys) = perform_gyo_prime_reduction(index_to_columns=index_to_columns)

    assert key_deletion_sequence is not None, 'The key deletion sequence is None'
    assert superset_key_of_deleted_keys is not None, 'The superset key of deleted keys is None'

    (G, root) = construct_join_tree(key_deletion_sequence=key_deletion_sequence, superset_key_of_deleted_keys=superset_key_of_deleted_keys)

    # Ensure that join tree is a tree
    assert nx.is_tree(G), 'The join tree is not a tree'

    # Ensure that the root node is part of the join tree
    assert root in G.nodes, 'The root node is not part of the join tree'

    # Ensure that each dataframe has a corresponding node in the join tree
    # This means that there are nodes [0, ..., len(index_to_columns) - 1] in the join tree
    for i in range(len(index_to_columns)):
        assert i in G.nodes, 'The join tree does not contain a node for dataframe {0}'.format(i)
    
    # Get set of all column names
    column_names = set()

    for i in range(len(index_to_columns)):
        column_names.update(index_to_columns[i])
    
    # Get mapping from column names to indices of dataframes
    column_name_to_index = {column_name: [] for column_name in column_names}

    for i in range(len(index_to_columns)):
        for column_name in index_to_columns[i]:
            column_name_to_index[column_name].append(i)

    # Ensure that the connectedness condition is satisified for every column name in the join tree
    # This means that for every pair of nodes (i, j) sharing a column name, there must be a path from i to j in the join tree such that the path only contains nodes sharing the column name
    for column_name in column_names:
        for i in column_name_to_index[column_name]:
            for j in column_name_to_index[column_name]:
                if i != j:
                    # Get all paths from i to j
                    paths = nx.all_simple_paths(G, source=i, target=j)

                    # Check that all paths only contain nodes sharing the column name
                    for path in paths:
                        for node in path:
                            assert column_name in index_to_columns[node], 'The connectedness condition is not satisfied for column name {0} in the join tree'.format(column_name)

if __name__ == '__main__':
    test_join_acyclic_query()
    test_join_tree_construction()