from typing import Optional, Tuple
from implementation.pd_join_optimizer import join_optimized, join_baseline, Strategy, VariableSelectionStrategy
from abc import ABC, abstractmethod
import json
import logging
import os
import pandas as pd
from queries import JoinQuery
from postgres_container import PostgresDatasetImporter, PostgresDockerContainer
import sqlalchemy
from sqlalchemy import text
import time
import scalene

class QueryExecutionTimer:
    def __init__(self):
        self.measurements = { 'starts': {}, 'ends': {} }

    def start(self, executor_name: str, query: JoinQuery):
        self.measurements['starts'][(executor_name, query)] = time.perf_counter_ns()

    def end(self, executor_name: str, query: JoinQuery):
        self.measurements['ends'][(executor_name, query)] = time.perf_counter_ns()

    def get_time_in_ms(self, executor_name: str, query: JoinQuery) -> float:
        return (self.measurements['ends'][(executor_name, query)] - self.measurements['starts'][(executor_name, query)]) / 1000000
    
    def get_times(self) -> pd.DataFrame:
        times = []
        for (executor_name, query), start in self.measurements['starts'].items():
            end = self.measurements['ends'][(executor_name, query)]
            times.append({
                'executor': executor_name,
                'dataset': query.dataset.get_name(),
                'query_id': query.query_id,
                'duration_ms': (end - start) / 1000000 
            })

        return pd.DataFrame(times)

# Given a join query, an executor will first call the preprocessing step of the join query to obtain a list of DataFrame objects.
# Then, it will call a function that implements the join operation and accepts a list of DataFrame objects as input. 
# There will be two different implementations for the join operation: one that utilizes optimizations from database theory and one naive baseline implementation that performs a sequence of two-way joins.
# The executor will then call the post-processing step of the join query to obtain the final result of the join query.
# As a last step, the executor will write the final result to a CSV file.
class JoinQueryExecutor(ABC):
    def __init__(self, output_directory: str, timer: QueryExecutionTimer):
        self.output_directory = output_directory
        self.timer = timer

    def execute(self, join_query: JoinQuery, strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        logging.info(f'Executing query {join_query.dataset.get_name()} {join_query.query_id} with {self.__class__.__name__}...')
        result, strategy = self.get_result(join_query, strategy)
        self.export_to_csv(result, f'{join_query.dataset.get_name()}_{join_query.query_id}.csv')

        return result, strategy
    
    @abstractmethod
    def get_result(self, join_query: JoinQuery, strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        pass

    def export_to_csv(self, dataframe: pd.DataFrame, file_name: str):
        dataframe.to_csv(os.path.join(self.output_directory, file_name), index=False)

class PythonJoinQueryExecutor(JoinQueryExecutor, ABC):
    def get_result(self, join_query: JoinQuery, strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        dataframes = join_query.preprocess()

        self.timer.start(self.__class__.__name__, join_query)
        scalene.scalene_profiler.start() if hasattr(scalene, 'scalene_profiler') else None
        dataframe, strategy = self.join(dataframes, strategy)
        scalene.scalene_profiler.stop() if hasattr(scalene, 'scalene_profiler') else None
        self.timer.end(self.__class__.__name__, join_query)

        logging.info(f'Join execution time: {self.timer.get_time_in_ms(self.__class__.__name__, join_query)} ms')

        dataframe = join_query.postprocess(dataframe)

        logging.debug(f'Final result – {dataframe.shape[0]} rows, columns: {dataframe.columns.tolist()}')
        
        return dataframe, strategy

    @abstractmethod
    def join(self, dataframes: list[pd.DataFrame], strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        pass

class BaselineQueryExecutor(PythonJoinQueryExecutor):
    def join(self, dataframes: list[pd.DataFrame], strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        return join_baseline(dataframes, strategy)

class OptimizedQueryExecutor(PythonJoinQueryExecutor):
    def __init__(self, 
                 output_directory: str, 
                 timer: QueryExecutionTimer, 
                 reduce_duplicates: bool = False,
                 variable_selection_strategy: VariableSelectionStrategy = VariableSelectionStrategy.LOWEST_DISTINCT_COUNT):
        super().__init__(output_directory, timer)
        self.reduce_duplicates = reduce_duplicates
        self.variable_selection_strategy = variable_selection_strategy

    def join(self, dataframes: list[pd.DataFrame], strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        return join_optimized(dataframes, 
                    strategy,
                    self.reduce_duplicates,
                    self.variable_selection_strategy)

class PostgresQueryExecutor(JoinQueryExecutor):
    def __init__(self, output_directory: str, timer: QueryExecutionTimer):
        super().__init__(output_directory, timer)

        # Load environment from JSON file `postgres_container.json`
        with open('postgres_container.json', 'r') as environment_file:
            environment = json.load(environment_file)

        container = PostgresDockerContainer(environment, data_path='../tmp/data')

        # Avoid re-importing datasets, if container is already running
        if not container.is_ready():
            container.start()
            container.wait_for_readiness()

            importer = PostgresDatasetImporter(container=container)
            importer.import_all_datasets()

        # Connect to Postgres database
        engine = sqlalchemy.create_engine(f"postgresql://{environment['POSTGRES_USER']}:{environment['POSTGRES_PASSWORD']}@{environment['POSTGRES_HOST']}:{environment['POSTGRES_PORT']}/{environment['POSTGRES_DB']}")
        self.connection = engine.connect()
    
    def get_result(self, join_query: JoinQuery, strategy: Strategy = None) -> Tuple[pd.DataFrame, Optional[Strategy]]:
        sql_query = join_query.get_sql_query()
        logging.debug(f'Executing SQL query:\n{sql_query}')
        self.timer.start(self.__class__.__name__, join_query)
        result = self.connection.execute(text(f'SET search_path TO {join_query.dataset.get_schema_name()}; {sql_query}'))
        dataframe = pd.DataFrame(result.fetchall(), columns=result.keys())
        self.timer.end(self.__class__.__name__, join_query)

        return dataframe, None
    
# Given two executors and a join query, the join query shall be executed by both executors and the results – i.e. Pandas DataFrame objects – shall be compared.
# The comparison should ignore the order of rows and columns.
# If the results are not equal, the cause for the inequality should be determined.
class JoinQueryExecutionComparator:
    def __init__(self, executor_1: JoinQueryExecutor, executor_2: JoinQueryExecutor, output_directory: str):
        self.executor_1 = executor_1
        self.executor_2 = executor_2
        self.output_directory = output_directory

    def compare(self, join_query: JoinQuery):
        logging.info(f'Comparing query {join_query.dataset.get_name()} {join_query.query_id} using executors ({self.executor_1.__class__.__name__}, {self.executor_2.__class__.__name__})...')
        result_1, _ = self.executor_1.execute(join_query)
        result_2, _ = self.executor_2.execute(join_query)

        # First, check if the column names are equal, while ignoring the order of columns
        # Also consider the case that column names could be duplicated in one of the DataFrames
        if sorted(result_1.columns.tolist()) != sorted(result_2.columns.tolist()):
            raise JoinQueryResultNotEqualException(f'Column names are not equal: {result_1.columns.tolist()} != {result_2.columns.tolist()}')
        
        # If there is a duplicate column name in one of the DataFrames, no meaningful comparison can be made
        if len(result_1.columns) != len(set(result_1.columns)):
            raise JoinQueryResultNotEqualException(f'Column names of {self.executor_1.__class__.__name__} result are not unique: {result_1.columns.tolist()}')
        elif len(result_2.columns) != len(set(result_2.columns)):
            raise JoinQueryResultNotEqualException(f'Column names of {self.executor_2.__class__.__name__} result are not unique: {result_2.columns.tolist()}')
        
        if result_1.empty and result_2.empty:
            logging.info(f'Results for execution of query {join_query.dataset.get_name()} {join_query.query_id} with executors ({self.executor_1.__class__.__name__}, {self.executor_2.__class__.__name__}) is equal')
            return
        
        result_1 = result_1.astype(object)
        result_2 = result_2.astype(object)

        # Ensure that column order is the same
        result_1 = result_1[result_2.columns]

        # Sort the DataFrames, because the order of rows should not matter for the comparison
        result_1 = result_1.sort_values(by=result_1.columns.tolist()).reset_index(drop=True)
        result_2 = result_2.sort_values(by=result_2.columns.tolist()).reset_index(drop=True)

        # If the two DataFrames are equal, return
        if result_1.equals(result_2):
            logging.info(f'Results for execution of query {join_query.dataset.get_name()} {join_query.query_id} with executors ({self.executor_1.__class__.__name__}, {self.executor_2.__class__.__name__}) is equal')
            return
        
        # Add a column with a cumulative count of duplicates to both DataFrames
        result_1['duplicate_counter'] = result_1.groupby(list(result_1.columns)).cumcount()
        result_2['duplicate_counter'] = result_2.groupby(list(result_2.columns)).cumcount()

        # Perform an outer join between the two DataFrames
        merged = result_1.merge(result_2, on=list(result_1.columns), how='outer', indicator=True)

        # Select the rows that are missing in either DataFrame
        missing = merged.loc[merged['_merge'] != 'both']

        # Export the missing rows to a CSV file
        missing.to_csv(f'{self.output_directory}/{join_query.dataset.get_name()}_{join_query.query_id}_comparison.csv', index=False)

        raise JoinQueryResultNotEqualException(f'Rows are not equal', missing)
    
# This exception is thrown, if the results of executing a join query with two different executors are not equal.
# Optionally, a DataFrame object can be provided that explains which rows exist in one result but not in the other.
class JoinQueryResultNotEqualException(Exception):
    def __init__(self, message: str, difference: pd.DataFrame = None):
        super().__init__(message)
        self.difference = difference

def get_all_executors() -> dict[str, type[JoinQueryExecutor]]:
    return {
        'postgres': PostgresQueryExecutor,
        'baseline': BaselineQueryExecutor,
        'optimized': OptimizedQueryExecutor
    }