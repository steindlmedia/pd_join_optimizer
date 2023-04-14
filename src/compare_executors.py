import logging
import logging.config
from queries import *
from datasets import *
from executors import *
import argparse
import sys
import json

with open('logging_configuration.json', 'r') as logging_configuration_file:
    logging.config.dictConfig(json.load(logging_configuration_file))

# Execute queries with two executors and compare the results.
def execute_queries_with_two_executors_and_compare(dataset_base_path, output_directory, executors=[JoinQueryExecutor]):
    if len(executors) != 2:
        raise Exception('You need to provide exactly two executors to compare the results.')
    
    queries = get_all_queries()
    datasets = get_all_datasets()
    
    comparator = JoinQueryExecutionComparator(executors[0], executors[1], output_directory=output_directory)

    for (dataset_name, queries) in queries.items():
        dataset = datasets[dataset_name](dataset_base_path)

        for (query_id, query) in queries.items():
            instance = query(dataset=dataset)

            try:
                comparator.compare(instance)
            except JoinQueryResultNotEqualException as e:
                logging.exception(e)

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Compare query execution between two executors.')
    parser.add_argument('--input-directory', help='Path to the dataset directory', required=True)
    parser.add_argument('--output-directory', help='Path to the output directory', required=True)
    parser.add_argument('--executors', help='List of executors to compare', nargs='+', choices=['postgres', 'optimized', 'baseline'], required=True)
    args = parser.parse_args()

    # Get arguments
    input_directory = args.input_directory
    output_directory = args.output_directory
    executors = args.executors

    # Check if we have exactly two executors
    if len(executors) != 2 or len(set(executors)) != 2:
        print('You need to provide exactly two different executors to compare the results.')
        sys.exit(1)
    
    # Get executor instances
    timer = QueryExecutionTimer()
    executor_1 = get_all_executors()[executors[0]](output_directory=output_directory, timer=timer)
    executor_2 = get_all_executors()[executors[1]](output_directory=output_directory, timer=timer)
    
    execute_queries_with_two_executors_and_compare(dataset_base_path=input_directory, 
                                                   output_directory=output_directory, 
                                                   executors=[executor_1, executor_2])