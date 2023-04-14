import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pd_join_optimizer import join_optimized, perform_gyo_prime_reduction
from test_util import assert_dataframes_equal

# Example from course "Advanced Database Systems"
def get_triangle_query_dataframes_with_skewed_data(m):
    a = pd.DataFrame({'a': [('a_' + str(i)) for i in range(m)]})
    a_0 = pd.DataFrame({'a': [('a_0') for i in range(m)]})
    b = pd.DataFrame({'b': [('b_' + str(i)) for i in range(m)]})
    b_0 = pd.DataFrame({'b': [('b_0') for i in range(m)]})
    c = pd.DataFrame({'c': [('c_' + str(i)) for i in range(m)]})
    c_0 = pd.DataFrame({'c': [('c_0') for i in range(m)]})

    R = pd.concat([a_0.merge(b, how='cross'), a.merge(b_0, how='cross')])
    S = pd.concat([b_0.merge(c, how='cross'), b.merge(c_0, how='cross')])
    T = pd.concat([a_0.merge(c, how='cross'), a.merge(c_0, how='cross')])

    return (R, S, T)

def test_gyo_reduction():
    index_to_columns = { 0: ['x_1', 'x_2'], 1: ['x_2', 'x_3'], 2: ['x_3', 'x_1'] }
    (key_deletion_sequence, superset_key_of_deleted_keys) = perform_gyo_prime_reduction(index_to_columns=index_to_columns)

    assert key_deletion_sequence is None, 'The key deletion sequence is not None'
    assert superset_key_of_deleted_keys is None, 'The superset key of deleted keys is not None'

def test_triangle_query_join():
    R, S, T = get_triangle_query_dataframes_with_skewed_data(m = 2)

    result, _ = join_optimized([R, S, T])
    ground_truth = R.merge(S, how='inner').merge(T, how='inner')

    assert_dataframes_equal(actual=result, expected=ground_truth)

if __name__ == '__main__':
    test_gyo_reduction()
    test_triangle_query_join()