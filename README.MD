# Pandas Join optimizer

## Build and runtime dependencies

The creation of a Conda environment with Python >= 3.9 is recommended.
The `requirements.txt` file in folder `src/implementation` contains the required dependencies for the `join_optimized` (utilizes optimizations known from database theory) and `join_baseline` functions.

### Evaluation

Additionally, the file `requirements.txt` in folder `src` provides additional dependencies needed for the evaluation and for verifying correctness of query execution results.

The retrieval and post-processing of datasets requires a Unix or Linux OS with the following applications installed:

* `sort`
* `tar`
* `gzip`
* `mv`
* `tail`
* `sort`
* `rm`
* `sed`

Furthermore, at least Java 8 JRE must be installed in order to generate the LUBM dataset.

Finally, a local Docker daemon is required (1) for performing post-processing on the IMDB dataset after retrieval, and (2) for comparing the results of a query execution using pandas to a PostgreSQL DB.

### Creation of Conda environment

The following shell script can be used to create a suitable Conda environment.

```sh
$ cd src
$ conda create -n pandas python=3.11.2 -y
$ conda activate pandas
$ pip3 install -r requirements.txt # requirements for evaluation
$ cd implementation
$ pip3 install -r requirements.txt
$ cd ..
```

## Get datasets

All datasets can be generated respectively retrieved from an external source with a single command into a specified output directory.

```
usage: get_datasets.py [-h] --directory DIRECTORY
```

## Perform evaluation

The evaluation consists of **10 runs** for each (query, executor) combination, but this can be changed with the optional `runs` argument.
Both the runtime and the peak memory usage of the join function execution are measured and stored into the file `results.csv` in the output directory.
In order to measure the peak memory usage, a query execution is repeated with the same strategy (i.e. join tree and associated pre-order/post-order traversals or partitioning for the `join_optimized` function; or just the join sequence if the `join_baseline` function was executed) as was used by the first execution. 
By default, the partitioning variable selection strategy `lowest-distinct-count` is used.
In order to be able to make a meaningful comparison of the evaluation results when trying out different parameter values, we use the number of the current iteration as random seed, so that the same random numbers are generated for the same iteration in the next evaluation run.

```
usage: perform_evaluation.py [-h] --input-directory INPUT_DIRECTORY --output-directory OUTPUT_DIRECTORY [--disable-reduce-duplicates]
                             [--variable-selection-strategy {random,lowest-distinct-count}] [--runs RUNS]
```

Under the hood, the evaluation calls the Python script `execute_query.py` in a subprocess. 

```
usage: execute_query.py [-h] --dataset DATASET --query QUERY --executor {postgres,optimized,baseline} --input-directory INPUT_DIRECTORY --output-directory OUTPUT_DIRECTORY [--reduce-duplicates]
                        [--variable-selection-strategy {random,lowest-distinct-count}] [--seed SEED]
```

Query execution results are stored into the specified output directory as CSV files.

## Compare results between two executors

It is possible to compare the results between different executors, e.g. the Python functions (`join_optimized`, `join_baseline`).
To verify correctness between different join implementations, it is also possible to compare the execution of the Python join functions with a PostgreSQL database.
In this case, SQL queries from `../sql_queries` are retrieved. 

```
usage: compare_executors.py [-h] --input-directory INPUT_DIRECTORY --output-directory OUTPUT_DIRECTORY --executors {postgres,optimized,baseline}
                            [{postgres,optimized,baseline} ...]
```

In case a difference has been detected, a file with suffix `_comparison.csv` is created in the specified output directory that enables a user to see the differences at a glance.

## Create PostgreSQL Docker container instance with datasets

If a query execution with executor `postgres` is triggered, we check if there is already a running PostgreSQL Docker container – based on the settings defined in the `postgres_container.json` file.
If this is not the case, a new instance is created and an import of all datasets is automatically performed.

Alternatively, it is also possible to start a PostgreSQL Docker container instance with interactive Python script `postgres_container.py`. In case no running instance exists, a user can either import only a specific dataset, or all datasets, into a new instance. 

```
usage: postgres_container.py [-h] --directory DIRECTORY
```

## Unit tests

Some unit tests have been implemented.
They can be executed with the `pytest` command in directory `src`.