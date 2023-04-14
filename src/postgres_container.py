import os
import docker
import time
import logging 
import json
import logging.config
import argparse

with open('logging_configuration.json', 'r') as logging_configuration_file:
    logging.config.dictConfig(json.load(logging_configuration_file))

class PostgresDockerContainer():
    def __init__(self, environment, data_path):
        self.environment = environment
        self.data_path = data_path

        # Get container, if is already running at the specified port
        self.container = self.get_container()

    def get_container(self):
        if 'POSTGRES_PORT' not in self.environment:
            return None
        
        client = docker.from_env()
        containers = client.containers.list(all=True, filters={'ancestor': 'postgres:latest', 'status': 'running'})
        for container in containers:
            for port in container.ports:
                for value in container.ports[port]:
                    if value.get('HostPort') == str(self.environment['POSTGRES_PORT']):
                        logging.info(f'Found running PostgreSQL container with ID {container.id} at port {self.environment["POSTGRES_PORT"]}.')
                        return container

        return None

    def is_ready(self):
        if self.container is None:
            return False
        
        result = self.container.exec_run(f'pg_isready -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]}', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD']})
        return result.exit_code == 0

    def wait_for_readiness(self):
        logging.info('Waiting for PostgreSQL container to be ready...')

        while True:
            if self.is_ready():
                logging.info('PostgreSQL container is ready.')
                break
            time.sleep(1)

    # Start a Docker container using the official PostgreSQL image with the given port exposed.
    def start(self):
        logging.info('Starting PostgreSQL container...')
        client = docker.from_env()
        ports = {}

        if 'POSTGRES_PORT' in self.environment:
            ports['5432'] = self.environment['POSTGRES_PORT']

        self.container = client.containers.run(
            "postgres:latest",
            environment=self.environment,
            ports=ports,
            volumes={
                os.path.abspath(self.data_path): {
                    'bind': '/data',
                    'mode': 'rw',
                }
            },
            shm_size="1G",
            detach=True)

        return self.container

    def import_sql_file(self, dataset_name, sql_file_name):
        logging.info(f"Importing SQL file '{dataset_name}/{sql_file_name}'...")
        result = self.container.exec_run(f'psql -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]} -f /data/{dataset_name}/{sql_file_name}', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD'], 'PGOPTIONS': self.environment['PGOPTIONS']})
        logging.debug(result)
    
    def create_schema_if_not_exists_and_switch_to_it(self, schema_name):
        logging.info(f"Creating schema '{schema_name}'...")
        command = f'CREATE SCHEMA IF NOT EXISTS {schema_name}'
        result = self.container.exec_run(f'psql -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]} -c "{command}"', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD']})
        logging.debug(result)

        logging.info(f"Switching to schema '{schema_name}'...")
        self.environment['PGOPTIONS'] = f'--search_path={schema_name}'

    def create_tables(self, table_name, column_names):
        logging.info(f"Creating table '{table_name}'...")
        command = f'CREATE TABLE {table_name} ({", ".join(map(lambda column_name: column_name + " varchar", column_names))})'
        result = self.container.exec_run(f'psql -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]} -c "{command}"', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD'], 'PGOPTIONS': self.environment['PGOPTIONS']})
        logging.debug(result)

    def import_from_file(self, dataset_name, file_name, table_name, column_names=None, has_header=False, escape_character=None, delimiter=','):
        logging.info(f"Importing file '{dataset_name}/{file_name}' into table '{table_name}'...")
        columns = f"({', '.join(column_names)})" if column_names is not None else ''
        options = ['FORMAT CSV']

        if has_header:
            options.append('HEADER')
        
        if escape_character is not None:
            options.append("ESCAPE '" + escape_character + "'")
        
        options.append("DELIMITER '" + delimiter + "'")
        
        command = f"COPY {table_name} {columns}FROM '/data/{dataset_name}/{file_name}' WITH ({', '.join(options)})"
        result = self.container.exec_run(f'psql -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]} -c "{command}"', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD'], 'PGOPTIONS': self.environment['PGOPTIONS']})
        logging.debug(result)

    def export_to_csv(self, dataset_name, table_name, with_header=True):
        logging.info(f"Exporting table '{table_name}' to .csv file...")
        options = ['FORMAT CSV']

        if with_header:
            options.append('HEADER')
        
        command = f"COPY {table_name} TO '/data/{dataset_name}/{table_name + '.csv'}' WITH ({', '.join(options)})"
        result = self.container.exec_run(f'psql -U {self.environment["POSTGRES_USER"]} -d {self.environment["POSTGRES_DB"]} -c "{command}"', environment={'PGPASSWORD': self.environment['POSTGRES_PASSWORD'], 'PGOPTIONS': self.environment['PGOPTIONS']})
        logging.debug(result)

    def stop(self):
        if self.container is None:
            logging.info('PostgreSQL container is not running.')
            return
        
        logging.info('Stopping PostgreSQL container...')
        self.container.stop()
        self.container.remove(v=True)
        logging.info('PostgreSQL container stopped.')
    
class PostgresDatasetExporter:
    def __init__(self, container: PostgresDockerContainer):
        self.container = container
        self.base_path = container.data_path

    def export(self, dataset_name, table_name):
        logging.info(f"Exporting table '{table_name}' from dataset '{dataset_name}'...")
        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.export_to_csv(dataset_name=dataset_name, table_name=table_name, with_header=False)
    
    def full_export(self, dataset_name):
        for csv_file in [file for file in os.listdir(os.path.join(self.base_path, dataset_name)) if file.endswith('.csv')]:
            self.export(dataset_name, csv_file[:-4])

class PostgresDatasetImporter:
    def __init__(self, container: PostgresDockerContainer):
        self.container = container
        self.base_path = container.data_path
        self.import_functions = {
            'LUBM': self.import_lubm_dataset, 
            'IMDB': self.import_imdb_dataset,
            'Facebook': self.import_facebook_dataset,
            'Arxiv': self.import_arxiv_dataset,
            'G+': self.import_gplus_dataset,
            'LiveJournal': self.import_livejournal_dataset,
            'Orkut': self.import_orkut_dataset,
            'Patents': self.import_patents_dataset
        }
    
    def import_lubm_dataset(self):
        dataset_name = 'lubm'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)

        for csv_file in [file for file in os.listdir(os.path.join(self.base_path, dataset_name)) if file.endswith('.csv')]:
            self.container.create_tables(table_name=csv_file[:-4], column_names=['subject', 'object'])
            self.container.import_from_file(dataset_name=dataset_name, file_name=csv_file, table_name=csv_file[:-4], column_names=['subject', 'object'])

    def import_imdb_dataset(self):
        dataset_name = 'imdb'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.import_sql_file(dataset_name, 'schematext.sql')

        for csv_file in [file for file in os.listdir(os.path.join(self.base_path, dataset_name)) if file.endswith('.csv')]:
            self.container.import_from_file(dataset_name=dataset_name, file_name=csv_file, table_name=csv_file[:-4], escape_character='\\')

    def import_facebook_dataset(self):
        table_name = 'facebook'
        dataset_name = 'facebook'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='facebook_combined.txt', table_name=table_name, column_names=['x', 'y'], delimiter=' ')

    def import_arxiv_dataset(self):
        table_name = 'arxiv'
        dataset_name = 'arxiv'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='ca-GrQc.txt', table_name=table_name, column_names=['x', 'y'], delimiter='\t')

    def import_gplus_dataset(self):
        table_name = 'gplus'
        dataset_name = 'gplus'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='gplus_combined.txt', table_name=table_name, column_names=['x', 'y'], has_header=False, delimiter=' ')

    def import_livejournal_dataset(self):
        table_name = 'livejournal'
        dataset_name = 'livejournal'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='soc-LiveJournal1.txt', table_name=table_name, column_names=['x', 'y'], has_header=False, delimiter='\t')

    def import_orkut_dataset(self):
        table_name = 'orkut'
        dataset_name = 'orkut'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='com-orkut.ungraph.txt', table_name=table_name, column_names=['x', 'y'], has_header=False, delimiter='\t')

    def import_patents_dataset(self):
        table_name = 'patents'
        dataset_name = 'patents'

        self.container.create_schema_if_not_exists_and_switch_to_it(schema_name=dataset_name)
        self.container.create_tables(table_name=table_name, column_names=['x', 'y'])
        self.container.import_from_file(dataset_name=dataset_name, file_name='cit-Patents.txt', table_name=table_name, column_names=['x', 'y'], has_header=False, delimiter='\t')

    def get_dataset_names(self):
        return self.import_functions.keys()

    def import_dataset(self, dataset_name: str):
        logging.info(f'Importing {dataset_name} dataset...')
        self.import_functions[dataset_name]()

    def import_all_datasets(self):
        logging.info('Importing all datasets...')
        for dataset in self.import_functions.keys():
            self.import_dataset(dataset)

if __name__ == '__main__':
    # Load environment from JSON file `postgres_container.json`
    with open('postgres_container.json', 'r') as environment_file:
        environment = json.load(environment_file)

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Start PostgreSQL Docker container.')
    parser.add_argument('--directory', help='Path to the dataset directory', required=True)
    args = parser.parse_args()

    container = PostgresDockerContainer(environment, data_path=args.directory)
    stop_container = True

    try:
        if container.is_ready():
            print('The container is already running.')
            stop_container = input('Do you want to stop the container? (y/n) ') == 'y'
            exit(0)
            
        container.start()
        container.wait_for_readiness()

        importer = PostgresDatasetImporter(container=container)
        datasets = list(importer.get_dataset_names())

        # To prevent tables with the same name being created in the same schema, the user currently needs to choose which dataset should be imported.
        print('Which dataset should be imported?')
        print('0. All datasets')

        for i, dataset in enumerate(datasets):
            print(f'{i + 1}. {dataset}')

        dataset = input('Enter the number of the dataset: ')

        if dataset == '0':
            importer.import_all_datasets()
        elif dataset not in list(map(lambda x: f'{x}', range(1, len(datasets) + 1))):
            print('Invalid dataset.')
            exit(1)
        else:
            importer.import_dataset(dataset_name=datasets[int(dataset) - 1])
        
        stop_container = input('Do you want to stop the container? (y/n) ') == 'y'
    finally:
        if stop_container:
            container.stop()