def assert_dataframes_equal(actual, expected):
    # Assert same columns
    assert sorted(actual.columns.tolist()) == sorted(expected.columns.tolist()), 'The columns of the result are not equal to the columns of the ground truth'

    # Ensure that column order is the same
    actual = actual[expected.columns]

    # Sort the DataFrames, because the order of rows should not matter for the comparison
    actual = actual.sort_values(by=actual.columns.tolist()).reset_index(drop=True)
    expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)

    if not actual.equals(expected):
        # Add a column with a cumulative count of duplicates to both DataFrames
        actual['duplicate_counter'] = actual.groupby(list(actual.columns)).cumcount()
        expected['duplicate_counter'] = expected.groupby(list(expected.columns)).cumcount()

        # Perform an outer join between the two DataFrames
        merged = actual.merge(expected, on=list(actual.columns), how='outer', indicator=True)

        # Select the rows that are missing in either DataFrame
        missing = merged.loc[merged['_merge'] != 'both']

        assert False, 'The result is not equal to the ground truth \n{})'.format(missing)