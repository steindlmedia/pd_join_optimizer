import os
import subprocess
import requests
from tqdm import tqdm
from abc import ABC, abstractmethod
import logging
import logging.config
import json
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser
from postgres_container import PostgresDockerContainer, PostgresDatasetImporter, PostgresDatasetExporter
import pandas as pd
import argparse

with open('logging_configuration.json', 'r') as logging_configuration_file:
    logging.config.dictConfig(json.load(logging_configuration_file))

class DatasetDownloader(ABC):
    def __init__(self, target_dir):
        self.target_dir = target_dir

    @abstractmethod
    def get_expected_filenames(self):
        return []
    
    def is_dataset_complete(self):
        expected_filenames = self.get_expected_filenames()

        # Check if all files are present
        filenames = [file for file in os.listdir(self.target_dir)]
        dataset_complete = True

        for filename in expected_filenames:
            if filename not in filenames:
                dataset_complete = False
                break

        return dataset_complete

    def download(self):
        if os.path.exists(self.target_dir):
            if self.is_dataset_complete():
                logging.info('Dataset already complete, skipping download')
                return
            else:
                # If the dataset is not complete, we re-generate it
                logging.info(f'Removing output folder {self.target_dir}')
                subprocess.run(['rm', '-rf', self.target_dir])
        
        logging.info(f'Creating output folder {self.target_dir}')
        os.makedirs(self.target_dir)

        self.redownload()

        # Perform post-processing, e.g. extract dataset
        self.postprocess(self.file_name)

        # Verify that the dataset is complete
        if not self.is_dataset_complete():
            raise Exception('The dataset is not complete after download + post-processing')

    def redownload(self):
        logging.info('Downloading dataset')
        self.download_file_with_progress(self.url, self.file_name)
        
    def remove_comments(self, file_name, number_of_comment_lines):
        logging.info('The following lines are removed from the file because they are comments:')
        subprocess.run(['head', '-n', f'{number_of_comment_lines}', file_name], cwd=self.target_dir)
        subprocess.run(f'tail -n +{number_of_comment_lines + 1} {file_name} > {file_name}.tmp', shell=True, cwd=self.target_dir)
        subprocess.run(['mv', file_name + '.tmp', file_name], cwd=self.target_dir)
        
    @abstractmethod
    def postprocess(self, file_name):
        pass
    
    def download_file_with_progress(self, url, file_name):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('Content-Length', 0))
        block_size = 1024
        wrote = 0
        file_path = os.path.join(self.target_dir, file_name)
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        logging.debug(f'Current working directory: {os.path.abspath(os.curdir)}')

        with open(file_path, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
                wrote += len(data)
        
        progress_bar.close()

        if total_size != 0 and wrote != total_size:
            raise Exception("An error occurred while downloading the file from " + url)

    def add_reverse_edges_to_file(self, file_name, sep):
        logging.info('Adding reverse edges to dataset')
        df = pd.read_csv(os.path.join(self.target_dir, file_name), sep=sep, skiprows=0, names=["x", "y"])
        df_reverse = df.copy(deep=True).rename(columns={"x": "y", "y": "x"})
        df = pd.concat([df, df_reverse])
        df.to_csv(os.path.join(self.target_dir, file_name), sep=sep, index=False, header=False)

class FacebookDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'https://snap.stanford.edu/data/facebook_combined.txt.gz'
        self.file_name = 'facebook_combined.txt.gz'

    def get_expected_filenames(self):
        return ['facebook_combined.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

        self.add_reverse_edges_to_file(self.get_expected_filenames()[0], sep=" ")

class ArxivDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'https://snap.stanford.edu/data/ca-GrQc.txt.gz'
        self.file_name = 'ca-GrQc.txt.gz'

    def get_expected_filenames(self):
        return ['ca-GrQc.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

        self.remove_comments(self.get_expected_filenames()[0], 4)

class GPlusDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'http://snap.stanford.edu/data/gplus_combined.txt.gz'
        self.file_name = 'gplus_combined.txt.gz'
    
    def get_expected_filenames(self):
        return ['gplus_combined.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

class LiveJournalDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz'
        self.file_name = 'soc-LiveJournal1.txt.gz'
    
    def get_expected_filenames(self):
        return ['soc-LiveJournal1.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

        self.remove_comments(self.get_expected_filenames()[0], 4)

class OrkutDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz'
        self.file_name = 'com-orkut.ungraph.txt.gz'
    
    def get_expected_filenames(self):
        return ['com-orkut.ungraph.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

        self.remove_comments(self.get_expected_filenames()[0], 4)
    
class PatentsDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)
        self.url = 'https://snap.stanford.edu/data/cit-Patents.txt.gz'
        self.file_name = 'cit-Patents.txt.gz'
    
    def get_expected_filenames(self):
        return ['cit-Patents.txt']
    
    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['gzip', '-d', file_name], cwd=self.target_dir)

        self.remove_comments(self.get_expected_filenames()[0], 4)

class IMDBDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir):
        super().__init__(target_dir)

        # The paper "How Good Are Query Optimizers, Really?" used the following dataset:
        # http://homepages.cwi.nl/~boncz/job/imdb.tgz
        self.url = 'http://homepages.cwi.nl/~boncz/job/imdb.tgz'
        self.file_name = 'imdb.tgz'
    
    def get_expected_filenames(self):
        return ['name.csv', 'movie_companies.csv', 'aka_name.csv', 'movie_info.csv', 'movie_keyword.csv', 'person_info.csv', 'comp_cast_type.csv', 'complete_cast.csv', 'char_name.csv', 'movie_link.csv', 'company_type.csv', 'cast_info.csv', 'info_type.csv', 'company_name.csv', 'aka_title.csv', 'kind_type.csv', 'role_type.csv', 'movie_info_idx.csv', 'keyword.csv', 'link_type.csv', 'title.csv']

    def postprocess(self, file_name):
        logging.info('Extracting dataset')
        subprocess.run(['tar', '-xzf', file_name], cwd=self.target_dir)

        logging.info('Removing .tgz file')
        subprocess.run(['rm', file_name], cwd=self.target_dir)

        # Some of the downloaded CSV files are not compatible with read_csv() in Pandas.
        # There are problems regarding the quotation and escape characters, e.g. if a " character appears in a column it should be escaped by another " character according to RFC-4180.
        # However, for example in the `company_name.csv` file there are lines containing \" instead of "".
        # This causes an error when trying to import the CSV file with Pandas.
        # I have tried adding the `quotechar='"'` and `escapechar='\\'` arguments to the `read_csv` function call. 
        # But, in this case the back slashes from column value `TBWA\CHIAT\DAY` were removed, which is not desired since this led to a deviation of query execution results later on.
        # Replacing \" with "" in the CSV files using `sed` also did not fix all issues.
        # As a result, I decided to use a PostgreSQL Docker container to import the CSV files into a database and then re-export them as CSV files.

        logging.info('Using PostgreSQL docker container to fix CSV files so that they can be imported with Pandas')
        self.fix_csv_files()

    def fix_csv_files(self):
        environment = {
            "POSTGRES_PASSWORD": "mysecret",
            "POSTGRES_USER": "postgres",
            "POSTGRES_DB": "postgres",
            "POSTGRES_HOST": "localhost",
        }

        dataset_name = os.path.basename(os.path.normpath(self.target_dir))
        data_path = os.path.abspath(os.path.join(self.target_dir, os.pardir))

        container = PostgresDockerContainer(environment, data_path=data_path)

        try:
            if not container.is_ready():
                container.start()
                container.wait_for_readiness()
            
            PostgresDatasetImporter(container).import_imdb_dataset()
            PostgresDatasetExporter(container).full_export(dataset_name=dataset_name)
        finally:
            container.stop()

class CSVPartitioningSink():
    def __init__(self, output_path):
        self.file_handlers = {}
        self.output_path = output_path

    def __del__(self):
        # Close all file handlers
        for p in self.file_handlers:
            self.file_handlers[p].close()

    def triple(self, s, p, o):
        # Remove base URI of ontology from subject, predicate and object
        s = s.replace('http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#', '')
        p = p.replace('http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#', '')
        o = o.replace('http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#', '')

        # Remove "http://www.w3.org/1999/02/22-rdf-syntax-ns#" from predicate
        p = p.replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', '')

        # Open file handler for predicate if not already open
        if p not in self.file_handlers:
            self.file_handlers[p] = open(os.path.join(self.output_path, p + '.csv'), 'w')
        
        # Write subject and object to file
        self.file_handlers[p].write(s + ',' + o + '\n')

class LUBMDatasetDownloader(DatasetDownloader):
    def __init__(self, target_dir, scaling_factor = 1000):
        super().__init__(target_dir)
        self.file_name = 'Universities-1.nt'
        self.scaling_factor = scaling_factor
    
    def get_expected_filenames(self):
        return ['name.csv', 'emailAddress.csv', 'telephone.csv', 'teachingAssistantOf.csv', 'memberOf.csv', 'headOf.csv', 'mastersDegreeFrom.csv', 'subOrganizationOf.csv', 'teacherOf.csv', 'worksFor.csv', 'undergraduateDegreeFrom.csv', 'takesCourse.csv', 'researchInterest.csv', 'advisor.csv', 'publicationAuthor.csv', 'type.csv', 'doctoralDegreeFrom.csv']

    def redownload(self):
        # Trigger the generation of the dataset in N-Triples format
        # File "Universities-1.nt" will be generated that contains 133 million triples, corresponding to 1000 universities
        logging.debug(f'Generating LUBM dataset in N-Triples format with {self.scaling_factor} universities')
        process = subprocess.Popen(['./generate.sh', '--quiet', '--timing', '-u', f'{self.scaling_factor}', '--format', 'NTRIPLES', '--consolidate', 'Full', '--threads', '1', '-o', os.path.abspath(os.path.join(os.getcwd(), self.target_dir))], cwd='./lubm-generator', stdout=subprocess.PIPE)
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                logging.debug(output.strip().decode())
        rc = process.poll()

    def postprocess(self, file_name):
        # After that we convert the N-Triples file to the CSV format.
        # This is done by using the "W3CNTriplesParser" class of the "rdflib" library and the "CSVPartitioningSink" class.
        # As part of this, vertical partitioning is also performed s.t. the triples are partitioned by predicate.
        # Hence, multiple CSV files are created, one for each predicate.
        nt_file_path = os.path.join(self.target_dir, file_name)

        # Before we can invoke the parser, we need to invoke the Unix/Linux application "sed" to remove all lines starting with <> from the N-Triples file.
        # This step ensures that the parser will not crash with the error "rdflib.exceptions.ParserError: Invalid line: .".
        logging.debug('Removing lines starting with <> from N-Triples file')
        subprocess.run(['sed', '-i', '', '/^<>/d', nt_file_path])

        logging.debug('Parsing N-Triples file and generating multiple CSV files')
        with open(nt_file_path, 'r') as f:
            sink = CSVPartitioningSink(output_path=self.target_dir)
            parser = W3CNTriplesParser(sink)
            parser.parse(f)

        # Remove duplicate lines in CSV file "type.csv"
        logging.debug('Removing duplicate lines in CSV file "type.csv"')
        subprocess.run(['sort', '-u', os.path.join(self.target_dir, 'type.csv'), '-o', os.path.join(self.target_dir, 'type.csv')])

        # Remove N-Triples file
        logging.debug(f'Removing N-Triples file {nt_file_path}')
        subprocess.run(['rm', nt_file_path])


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Get datasets.')
    parser.add_argument('--directory', help='Path to the dataset directory', required=True)
    args = parser.parse_args()

    # Get arguments
    directory = args.directory
    
    FacebookDatasetDownloader(os.path.join(directory, 'facebook')).download()
    ArxivDatasetDownloader(os.path.join(directory, 'arxiv')).download()
    GPlusDatasetDownloader(os.path.join(directory, 'gplus')).download()
    LiveJournalDatasetDownloader(os.path.join(directory, 'livejournal')).download()
    OrkutDatasetDownloader(os.path.join(directory, 'orkut')).download()
    PatentsDatasetDownloader(os.path.join(directory, 'patents')).download()
    IMDBDatasetDownloader(os.path.join(directory, 'imdb')).download()
    LUBMDatasetDownloader(os.path.join(directory, 'lubm')).download()