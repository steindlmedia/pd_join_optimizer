import os
import sys
import argparse
import pickle
import scalene
from datasets import get_all_datasets
from executors import QueryExecutionTimer, get_all_executors
from queries import get_all_queries
import random

# Create an argument parser
parser = argparse.ArgumentParser(description='Execute a query.')
parser.add_argument('--dataset', help='Name of the dataset', required=True)
parser.add_argument('--query', help='ID of the query', required=True)
parser.add_argument('--executor', help='Name of the executor', choices=['postgres', 'optimized', 'baseline'], required=True)
parser.add_argument('--input-directory', help='Path to the dataset directory', required=True)
parser.add_argument('--output-directory', help='Path to the output directory', required=True)
parser.add_argument('--reduce-duplicates', help='Reduce duplicates', action='store_true', default=False)
parser.add_argument('--variable-selection-strategy', help='Partitioning variable selection strategy', choices=['random', 'lowest-distinct-count'], default='lowest-distinct-count')
parser.add_argument('--seed', help='Random seed', type=int, default=None)

args = parser.parse_args()

# Get arguments
dataset_name = args.dataset
query_id = args.query
executor_name = args.executor
input_directory = args.input_directory
output_directory = args.output_directory
reduce_duplicates = args.reduce_duplicates
variable_selection_strategy = 2 if args.variable_selection_strategy == 'lowest-distinct-count' else 1

# Set random seed, if provided
if args.seed is not None:
    random.seed(args.seed)

timer = QueryExecutionTimer()

def load_objects(dataset_name, query_id, executor_name, input_directory, output_directory):
    try:
        dataset = get_all_datasets()[dataset_name](base_path=input_directory)
    except KeyError as e:
        print(f'Unknown dataset {dataset_name}.')
        sys.exit(1)

    try:
        query = get_all_queries()[dataset_name][query_id](dataset=dataset)
    except KeyError as e:
        print(f'Unknown query {query_id} for dataset {dataset_name}.')
        sys.exit(1)

    try:
        if executor_name == 'optimized':
            executor = get_all_executors()[executor_name](output_directory=output_directory, timer=timer, variable_selection_strategy=variable_selection_strategy, reduce_duplicates=reduce_duplicates)
        else:
            executor = get_all_executors()[executor_name](output_directory=output_directory, timer=timer)
    except KeyError as e:
        print(f'Unknown executor {executor_name}.')
        sys.exit(1)
    
    return dataset, query, executor

dataset, query, executor = load_objects(dataset_name, query_id, executor_name, input_directory=input_directory, output_directory=output_directory)

# If we are profiling, we load the strategy from the pickle file created by the previous run
if hasattr(scalene, 'scalene_profiler'): 
    with open(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_strategy.pickle', 'rb') as f:
        strategy = pickle.load(f)
else:
    strategy = None

result, strategy = executor.execute(query, strategy)

# If we are not profiling, we save the strategy to a pickle file and export the execution time as CSV
if not hasattr(scalene, 'scalene_profiler'):
    with open(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_strategy.pickle', 'wb') as f:
        pickle.dump(strategy, f)
    
    timer.get_times().to_csv(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_times.csv', index=False)
else:
    # Clean-up the pickle file, if we are profiling
    os.remove(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_strategy.pickle')