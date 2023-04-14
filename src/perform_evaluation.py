import subprocess
import json
from queries import *
from executors import *
import argparse
import numpy as np

# Create an argument parser
parser = argparse.ArgumentParser(description='Perform evaluation.')
parser.add_argument('--input-directory', help='Path to the dataset directory', required=True)
parser.add_argument('--output-directory', help='Path to the output directory', required=True)
parser.add_argument('--disable-reduce-duplicates', help='Disable reduction of duplicates', action='store_true', default=False)
parser.add_argument('--variable-selection-strategy', help='Partitioning variable selection strategy', choices=['random', 'lowest-distinct-count'], default='lowest-distinct-count')
parser.add_argument('--runs', help='Number of runs', type=int, default=10)

args = parser.parse_args()

input_directory = args.input_directory
output_directory = args.output_directory
variable_selection_strategy = args.variable_selection_strategy
reduce_duplicates = not args.disable_reduce_duplicates # for triangle queries, we usually want no duplicates – hence we reduce them by default
runs = args.runs

# Start Python script "execute_query.py" with the given arguments.
# We then wait for the process to terminate.
# In case of a non-zero return code, we raise an exception.
def execute_query(dataset: str, query: str, executor: str, input_directory, output_directory, seed = None, enable_profiling = False):
    command = ['scalene', '--off', '--cli', '--json', '--outfile', f'{output_directory}/{dataset}_{query}_{executor}_scalene.json'] if enable_profiling else []
    command += ['python'] if not enable_profiling else []
    command += ['execute_query.py', '--dataset', dataset, '--query', query, '--executor', executor, '--input-directory', input_directory, '--output-directory', output_directory, '--variable-selection-strategy', variable_selection_strategy]
    command += ['--reduce-duplicates'] if reduce_duplicates else []
    command += ['--seed', str(seed)] if seed is not None else []
    
    print(f'Executing command: {" ".join(command)}')

    # Start the subprocess.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read continously from stdout and stderr until the process terminates.
    while process.poll() is None:
        # Print the output of the subprocess.
        line = process.stdout.readline()
        if line:
            print(line.decode('utf-8').strip())
        line = process.stderr.readline()
        if line:
            print(line.decode('utf-8').strip())
    
    # Wait for the process to terminate.
    process.wait()

    if process.returncode != 0:
        raise Exception(f'The process terminated with non-zero return code {process.returncode}.')

queries = get_all_queries()
executors = list(filter(lambda executor: executor[0] != 'postgres', get_all_executors().items()))
results = pd.DataFrame(columns=['executor', 'dataset', 'query_id',  'duration_ms', 'peak_memory_usage_mb'])

for (dataset_name, queries) in queries.items():
    for (query_id, _) in queries.items():
        for (executor_name, _) in executors:
            for i in range(runs):
                print(f'Execution {i+1}')

                try: 
                    # First, we run the query without attached profiler in order to measure the execution time.
                    execute_query(dataset=dataset_name, query=query_id, executor=executor_name, input_directory=input_directory, output_directory=output_directory, seed=i)

                    # Second, we read the execution time from the CSV file.
                    with open(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_times.csv', 'r') as times_file:
                        result = pd.read_csv(times_file, index_col=0).iloc[0].to_dict()

                    # Third, we run the query again with a profiler attached such that we can measure the peak memory usage.
                    execute_query(dataset=dataset_name, query=query_id, executor=executor_name, input_directory=input_directory, output_directory=output_directory, seed=i, enable_profiling=True)

                    # Extract peak memory usage from the profiler JSON report.
                    # We consider only measurements for lines of the "pd_join_optimizer.py" file.
                    # This is done in this way to avoid reporting the memory usage prior to filtering of DataFrame objects, which would cause the reported memory usage to be too high.
                    with open(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_scalene.json', 'r') as scalene_report_file:
                        scalene_report = json.load(scalene_report_file)
                        peak_memory_usage = 0

                        if 'files' in scalene_report:
                            for filename, value in scalene_report['files'].items():
                                if not filename.endswith('pd_join_optimizer.py'):
                                    continue
                                    
                                for line in value['lines']:
                                    if line['n_peak_mb'] > peak_memory_usage:
                                        peak_memory_usage = line['n_peak_mb']

                    # Add peak memory usage to the result.
                    result['peak_memory_usage_mb'] = peak_memory_usage
                
                    # Add the newest result.
                    results.loc[len(results)] = [executor_name, dataset_name, query_id, result['duration_ms'], result['peak_memory_usage_mb']]

                    # Remove CSV file.
                    os.remove(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_times.csv')

                    # Remove profiler JSON report.
                    os.remove(f'{output_directory}/{dataset_name}_{query_id}_{executor_name}_scalene.json')
                except Exception as e:
                    logging.error(f'Failed to execute query {query_id} on dataset {dataset_name} with executor {executor_name}.')
                    logging.exception(e)

                    results.loc[len(results)] = [executor_name, dataset_name, query_id, np.nan, np.nan]

            # Store the results in CSV files.
            results.to_csv(f'{output_directory}/results.csv', index=False)