from abc import ABC, abstractmethod
import logging
import os
from datasets import *
import pandas as pd

# For each join query, we first need to load the required DataFrame objects from the dataset.
# Then, we need to perform some preprocessing steps s.t. we can perform a natural join later on. 
# This includes filtering DataFrames depending on the WHERE clause defined in the original SQL query, renaming columns depending on the join conditions (e.g. `id` column of one DataFrame is renamed to `movie_id` s.t. it matches the column name of another DataFrame), renaming columns to avoid that a join is performed and dropping columns that are neither required for the join operation nor used for the final result. 
# The preprocessing stage returns a list of DataFrame objects.
# A join query also has a post-processing step that will be invoked after the join operation has been performed.
# This post-processing step involves dropping columns that are not required for the final result, performing column renaming, and applying aggregations if necessary.
# The post-processing step returns a DataFrame object which corresponds to the final result of the join query.
class JoinQuery(ABC):
    def __init__(self, dataset: Dataset, query_id: str, path_to_sql_file: str):
        self.dataset = dataset
        self.query_id = query_id
        self.path_to_sql_file = path_to_sql_file
        self.sql_query = None

    @abstractmethod
    def preprocess(self) -> list[pd.DataFrame]:
        pass

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        return result

    def get_sql_query(self) -> str:
        if self.sql_query is None:
            if not self.is_aggregation_enabled():
                # Adapt path to SQL file in case no aggregation should be performed.
                # Use "without_aggregation" subdirectory
                path = os.path.join(os.path.dirname(self.path_to_sql_file), 'without_aggregation', os.path.basename(self.path_to_sql_file))

                if os.path.exists(path):
                    self.path_to_sql_file = path
                else:
                    logging.info(f'No SQL file without aggregation found at {path}. Using original SQL file.')
            
            with open(self.path_to_sql_file, 'r') as sql_file:
                self.sql_query = sql_file.read()

        return self.sql_query
    
    def is_aggregation_enabled(self) -> bool:
        # Disable aggregation due to different aggregation results on same data between Postgres and Pandas.
        return False

# IMDB: 2a
# SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code = '[de]' AND k.keyword = 'character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
class IMDB_2a(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '2a', '../sql_queries/imdb/2a.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[de]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)

        return [company_name, keyword, movie_companies, movie_keyword, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'movie_title': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'title': 'movie_title'}, inplace=True)

        return result
    
# IMDB: 2b
# SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code = '[nl]' AND k.keyword = 'character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
class IMDB_2b(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '2b', '../sql_queries/imdb/2b.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[nl]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)

        return [company_name, keyword, movie_companies, movie_keyword, title]
    
    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'movie_title': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'title': 'movie_title'}, inplace=True)

        return result

# IMDB: 2c
# SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code = '[sm]' AND k.keyword = 'character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
class IMDB_2c(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '2c', '../sql_queries/imdb/2c.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[sm]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)

        return [company_name, keyword, movie_companies, movie_keyword, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'movie_title': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'title': 'movie_title'}, inplace=True)

        return result

# IMDB: 2d
# SELECT MIN(t.title) AS movie_title FROM company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, title AS t WHERE cn.country_code = '[us]' AND k.keyword = 'character-name-in-title' AND cn.id = mc.company_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mc.movie_id = mk.movie_id;
class IMDB_2d(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '2d', '../sql_queries/imdb/2d.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)

        return [company_name, keyword, movie_companies, movie_keyword, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'movie_title': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'title': 'movie_title'}, inplace=True)

        return result

# IMDB: 4c
# SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS movie_title FROM info_type AS it, keyword AS k, movie_info_idx AS mi_idx, movie_keyword AS mk, title AS t WHERE it.info = 'rating' AND k.keyword like '%sequel%' AND mi_idx.info > '2.0' AND t.production_year > 1990 AND t.id = mi_idx.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND it.id = mi_idx.info_type_id;
class IMDB_4c(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '4c', '../sql_queries/imdb/4c.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        info_type = self.dataset.load_info_type()
        keyword = self.dataset.load_keyword()
        movie_info_idx = self.dataset.load_movie_info_idx()
        movie_keyword = self.dataset.load_movie_keyword()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        info_type = info_type[info_type['info'] == 'rating']
        keyword = keyword[keyword['keyword'].str.contains('sequel', na=False, regex=False)]
        movie_info_idx = movie_info_idx[movie_info_idx['info'] > '2.0']
        title = title[title['production_year'] > 1990]

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        info_type.drop(columns=['info'], inplace=True)
        keyword.drop(columns=['phonetic_code', 'keyword'], inplace=True)
        movie_info_idx.drop(columns=['id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum', 'production_year'], inplace=True)

        # Rename columns such that a natural join can be performed
        info_type.rename(columns={'id': 'info_type_id'}, inplace=True)
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)

        return [info_type, keyword, movie_info_idx, movie_keyword, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['info', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'rating': [result['info'].min()], 'movie_title': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'info': 'rating', 'title': 'movie_title'}, inplace=True)

        return result

# IMDB: 8c
# SELECT MIN(a1.name) AS writer_pseudo_name, MIN(t.title) AS movie_title FROM aka_name AS a1, cast_info AS ci, company_name AS cn, movie_companies AS mc, name AS n1, role_type AS rt, title AS t WHERE cn.country_code = '[us]' AND rt.role = 'writer' AND a1.person_id = n1.id AND n1.id = ci.person_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.role_id = rt.id AND a1.person_id = ci.person_id AND ci.movie_id = mc.movie_id;    
class IMDB_8c(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '8c', '../sql_queries/imdb/8c.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        aka_name = self.dataset.load_aka_name()
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        movie_companies = self.dataset.load_movie_companies()
        name = self.dataset.load_name()
        role_type = self.dataset.load_role_type()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        role_type = role_type[role_type['role'] == 'writer']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        aka_name.drop(columns=['id', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        cast_info.drop(columns=['id', 'person_role_id', 'note', 'nr_order'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        name.drop(columns=['name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        role_type.drop(columns=[], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        role_type.rename(columns={'id': 'role_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        name.rename(columns={'id': 'person_id'}, inplace=True)

        return [aka_name, cast_info, company_name, movie_companies, name, role_type, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'writer_pseudo_name': [result['name'].min()], 'movie_title': [result['title'].min()]})
            logging.debug(result)
        else:
            # Just perform renaming
            result.rename(columns={'name': 'writer_pseudo_name', 'title': 'movie_title'}, inplace=True)
        
        return result

# IMDB: 8d
# SELECT MIN(an1.name) AS costume_designer_pseudo, MIN(t.title) AS movie_with_costumes FROM aka_name AS an1, cast_info AS ci, company_name AS cn, movie_companies AS mc, name AS n1, role_type AS rt, title AS t WHERE cn.country_code = '[us]' AND rt.role = 'costume designer' AND an1.person_id = n1.id AND n1.id = ci.person_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.role_id = rt.id AND an1.person_id = ci.person_id AND ci.movie_id = mc.movie_id;
class IMDB_8d(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '8d', '../sql_queries/imdb/8d.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        aka_name = self.dataset.load_aka_name()
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        movie_companies = self.dataset.load_movie_companies()
        name = self.dataset.load_name()
        role_type = self.dataset.load_role_type()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        role_type = role_type[role_type['role'] == 'costume designer']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        aka_name.drop(columns=['id', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        cast_info.drop(columns=['id', 'person_role_id', 'note', 'nr_order'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum', 'country_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        name.drop(columns=['name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        role_type.drop(columns=['role'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        role_type.rename(columns={'id': 'role_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        name.rename(columns={'id': 'person_id'}, inplace=True)

        return [aka_name, cast_info, company_name, movie_companies, name, role_type, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'costume_designer_pseudo': [result['name'].min()], 'movie_with_costumes': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'costume_designer_pseudo', 'title': 'movie_with_costumes'}, inplace=True)
        
        return result

# IMDB: 10b
# SELECT MIN(chn.name) AS character, MIN(t.title) AS russian_mov_with_actor_producer FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note like '%(producer)%' AND cn.country_code = '[ru]' AND rt.role = 'actor' AND t.production_year > 2010 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;
class IMDB_10b(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '10b', '../sql_queries/imdb/10b.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        char_name = self.dataset.load_char_name()
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        company_type = self.dataset.load_company_type()
        movie_companies = self.dataset.load_movie_companies()
        role_type = self.dataset.load_role_type()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        role_type = role_type[role_type['role'] == 'actor']
        company_name = company_name[company_name['country_code'] == '[ru]']
        title = title[title['production_year'] > 2010]
        cast_info = cast_info[cast_info['note'].str.contains('(producer)', na=False, regex=False)]

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        char_name.drop(columns=['imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        cast_info.drop(columns=['id', 'person_id', 'nr_order'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        company_type.drop(columns=['kind'], inplace=True)
        movie_companies.drop(columns=['id', 'note'], inplace=True)
        role_type.drop(columns=[], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        company_type.rename(columns={'id': 'company_type_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        role_type.rename(columns={'id': 'role_id'}, inplace=True)
        char_name.rename(columns={'id': 'person_role_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)

        return [char_name, cast_info, company_name, company_type, movie_companies, role_type, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'character': [result['name'].min()], 'russian_mov_with_actor_producer': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'character', 'title': 'russian_mov_with_actor_producer'}, inplace=True)
        
        return result

# IMDB: 10c
# SELECT MIN(chn.name) AS character, MIN(t.title) AS movie_with_american_producer FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note like '%(producer)%' AND cn.country_code = '[us]' AND t.production_year > 1990 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;
class IMDB_10c(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '10c', '../sql_queries/imdb/10c.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        char_name = self.dataset.load_char_name()
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        company_type = self.dataset.load_company_type()
        movie_companies = self.dataset.load_movie_companies()
        role_type = self.dataset.load_role_type()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        title = title[title['production_year'] > 1990]
        cast_info = cast_info[cast_info['note'].str.contains('(producer)', na=False, regex=False)]

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        char_name.drop(columns=['imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        cast_info.drop(columns=['id', 'person_id', 'nr_order'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        company_type.drop(columns=['kind'], inplace=True)
        movie_companies.drop(columns=['id', 'note'], inplace=True)
        role_type.drop(columns=['role'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        company_type.rename(columns={'id': 'company_type_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        role_type.rename(columns={'id': 'role_id'}, inplace=True)
        char_name.rename(columns={'id': 'person_role_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)

        return [char_name, cast_info, company_name, company_type, movie_companies, role_type, title]
        
    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'character': [result['name'].min()], 'movie_with_american_producer': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'character', 'title': 'movie_with_american_producer'}, inplace=True)
        
        return result

# IMDB: 13a
# SELECT MIN(mi.info) AS release_date, MIN(miidx.info) AS rating, MIN(t.title) AS german_movie FROM company_name AS cn, company_type AS ct, info_type AS it, info_type AS it2, kind_type AS kt, movie_companies AS mc, movie_info AS mi, movie_info_idx AS miidx, title AS t WHERE cn.country_code = '[de]' AND ct.kind = 'production companies' AND it.info = 'rating' AND it2.info = 'release dates' AND kt.kind = 'movie' AND mi.movie_id = t.id AND it2.id = mi.info_type_id AND kt.id = t.kind_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND miidx.movie_id = t.id AND it.id = miidx.info_type_id AND mi.movie_id = miidx.movie_id AND mi.movie_id = mc.movie_id AND miidx.movie_id = mc.movie_id;
class IMDB_13a(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '13a', '../sql_queries/imdb/13a.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        company_type = self.dataset.load_company_type()
        info_type = self.dataset.load_info_type()
        info_type2 = info_type.copy(deep=True)
        kind_type = self.dataset.load_kind_type()
        movie_companies = self.dataset.load_movie_companies()
        movie_info = self.dataset.load_movie_info()
        movie_info_idx = self.dataset.load_movie_info_idx()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        kind_type = kind_type[kind_type['kind'] == 'movie']
        info_type = info_type[info_type['info'] == 'rating']
        info_type2 = info_type2[info_type2['info'] == 'release dates']
        company_name = company_name[company_name['country_code'] == '[de]']
        company_type = company_type[company_type['kind'] == 'production companies']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum', 'country_code'], inplace=True)
        company_type.drop(columns=['kind'], inplace=True)
        info_type.drop(columns=['info'], inplace=True)
        info_type2.drop(columns=['info'], inplace=True)
        kind_type.drop(columns=['kind'], inplace=True)
        movie_companies.drop(columns=['id', 'note'], inplace=True)
        movie_info.drop(columns=['id', 'note'], inplace=True)
        movie_info_idx.drop(columns=['id', 'note'], inplace=True)
        title.drop(columns=['imdb_index', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        info_type.rename(columns={'id': 'info_type_id'}, inplace=True)
        info_type2.rename(columns={'id': 'info_type_id2'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_type.rename(columns={'id': 'company_type_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        kind_type.rename(columns={'id': 'kind_id'}, inplace=True)
        movie_info.rename(columns={'info_type_id': 'info_type_id2', 'info': 'info2'}, inplace=True) # ensure join with info_type2 is performed, avoid name clash with movie_info_idx

        return [company_name, company_type, info_type, info_type2, kind_type, movie_companies, movie_info, movie_info_idx, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['info', 'info2', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'release_date': [result['info2'].min()], 'rating': [result['info'].min()], 'german_movie': [result['title'].min()]}) 
        else:
            # Just perform renaming
            result.rename(columns={'info2': 'release_date', 'info': 'rating', 'title': 'german_movie'}, inplace=True)
        
        return result

# IMDB: 13d
# SELECT MIN(cn.name) AS producing_company, MIN(miidx.info) AS rating, MIN(t.title) AS movie FROM company_name AS cn, company_type AS ct, info_type AS it, info_type AS it2, kind_type AS kt, movie_companies AS mc, movie_info AS mi, movie_info_idx AS miidx, title AS t WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND it.info = 'rating' AND it2.info = 'release dates' AND kt.kind = 'movie' AND mi.movie_id = t.id AND it2.id = mi.info_type_id AND kt.id = t.kind_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND miidx.movie_id = t.id AND it.id = miidx.info_type_id AND mi.movie_id = miidx.movie_id AND mi.movie_id = mc.movie_id AND miidx.movie_id = mc.movie_id;
class IMDB_13d(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '13d', '../sql_queries/imdb/13d.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        company_name = self.dataset.load_company_name()
        company_type = self.dataset.load_company_type()
        info_type = self.dataset.load_info_type()
        info_type2 = info_type.copy(deep=True)
        kind_type = self.dataset.load_kind_type()
        movie_companies = self.dataset.load_movie_companies()
        movie_info = self.dataset.load_movie_info()
        movie_info_idx = self.dataset.load_movie_info_idx()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        kind_type = kind_type[kind_type['kind'] == 'movie']
        info_type = info_type[info_type['info'] == 'rating']
        info_type2 = info_type2[info_type2['info'] == 'release dates']
        company_name = company_name[company_name['country_code'] == '[us]']
        company_type = company_type[company_type['kind'] == 'production companies']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        company_name.drop(columns=['imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum', 'country_code'], inplace=True)
        company_type.drop(columns=['kind'], inplace=True)
        info_type.drop(columns=['info'], inplace=True)
        info_type2.drop(columns=['info'], inplace=True)
        kind_type.drop(columns=['kind'], inplace=True)
        movie_companies.drop(columns=['id', 'note'], inplace=True)
        movie_info.drop(columns=['id', 'note', 'info'], inplace=True)
        movie_info_idx.drop(columns=['id', 'note'], inplace=True)
        title.drop(columns=['imdb_index', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        info_type.rename(columns={'id': 'info_type_id'}, inplace=True)
        info_type2.rename(columns={'id': 'info_type_id2'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        company_type.rename(columns={'id': 'company_type_id'}, inplace=True)
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        kind_type.rename(columns={'id': 'kind_id'}, inplace=True)
        movie_info.rename(columns={'info_type_id': 'info_type_id2'}, inplace=True) # ensure join with info_type2 is performed

        return [company_name, company_type, info_type, info_type2, kind_type, movie_companies, movie_info, movie_info_idx, title]
    
    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'info', 'title'])), inplace=True)

        if self.is_aggregation_enabled():        
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'producing_company': [result['name'].min()], 'rating': [result['info'].min()], 'movie': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'producing_company', 'info': 'rating', 'title': 'movie'}, inplace=True)

        return result

# IMDB: 16b
# SELECT MIN(an.name) AS cool_actor_pseudonym, MIN(t.title) AS series_named_after_char FROM aka_name AS an, cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE cn.country_code = '[us]' AND k.keyword = 'character-name-in-title' AND an.person_id = n.id AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND an.person_id = ci.person_id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;
class IMDB_16b(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '16b', '../sql_queries/imdb/16b.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        aka_name = self.dataset.load_aka_name()
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        name = self.dataset.load_name()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        aka_name.drop(columns=['id', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        cast_info.drop(columns=['id', 'person_role_id', 'note', 'nr_order', 'role_id'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        name.drop(columns=['name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        title.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        name.rename(columns={'id': 'person_id'}, inplace=True)

        return [aka_name, cast_info, company_name, keyword, movie_companies, movie_keyword, name, title]
    
    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name', 'title'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'cool_actor_pseudonym': [result['name'].min()], 'series_named_after_char': [result['title'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'cool_actor_pseudonym', 'title': 'series_named_after_char'}, inplace=True)

        return result

# IMDB: 17e
# SELECT MIN(n.name) AS member_in_charnamed_movie FROM cast_info AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t WHERE cn.country_code = '[us]' AND k.keyword = 'character-name-in-title' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;
class IMDB_17e(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '17e', '../sql_queries/imdb/17e.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        cast_info = self.dataset.load_cast_info()
        company_name = self.dataset.load_company_name()
        keyword = self.dataset.load_keyword()
        movie_companies = self.dataset.load_movie_companies()
        movie_keyword = self.dataset.load_movie_keyword()
        name = self.dataset.load_name()
        title = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        company_name = company_name[company_name['country_code'] == '[us]']
        keyword = keyword[keyword['keyword'] == 'character-name-in-title']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        cast_info.drop(columns=['id', 'person_role_id', 'note', 'nr_order', 'role_id'], inplace=True)
        company_name.drop(columns=['name', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], inplace=True)
        keyword.drop(columns=['phonetic_code'], inplace=True)
        movie_companies.drop(columns=['id', 'company_type_id', 'note'], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        name.drop(columns=['imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], inplace=True)
        title.drop(columns=['title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        company_name.rename(columns={'id': 'company_id'}, inplace=True)
        title.rename(columns={'id': 'movie_id'}, inplace=True)
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)
        name.rename(columns={'id': 'person_id'}, inplace=True)

        return [cast_info, company_name, keyword, movie_companies, movie_keyword, name, title]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['name'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'member_in_charnamed_movie': [result['name'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'name': 'member_in_charnamed_movie'}, inplace=True)
        
        return result

# IMDB: 32a
# SELECT MIN(lt.link) AS link_type, MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie FROM keyword AS k, link_type AS lt, movie_keyword AS mk, movie_link AS ml, title AS t1, title AS t2 WHERE k.keyword = '10,000-mile-club' AND mk.keyword_id = k.id AND t1.id = mk.movie_id AND ml.movie_id = t1.id AND ml.linked_movie_id = t2.id AND lt.id = ml.link_type_id AND mk.movie_id = t1.id;
class IMDB_32a(JoinQuery):
    def __init__(self, dataset: IMDBDataset):
        super().__init__(dataset, '32a', '../sql_queries/imdb/32a.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        keyword = self.dataset.load_keyword()
        link_type = self.dataset.load_link_type()
        movie_keyword = self.dataset.load_movie_keyword()
        movie_link = self.dataset.load_movie_link()
        title1 = self.dataset.load_title()
        title2 = self.dataset.load_title()

        # Apply filter
        logging.debug('Applying filter...')
        keyword = keyword[keyword['keyword'] == '10,000-mile-club']

        # For each DataFrame, drop the columns that are neither used for the join nor the result
        keyword.drop(columns=['phonetic_code'], inplace=True)
        link_type.drop(columns=[], inplace=True)
        movie_keyword.drop(columns=['id'], inplace=True)
        movie_link.drop(columns=['id'], inplace=True)
        title1.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)
        title2.drop(columns=['imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], inplace=True)

        # Rename columns such that a natural join can be performed
        title1.rename(columns={'id': 'movie_id'}, inplace=True)
        title2.rename(columns={'id': 'linked_movie_id', 'title': 'title2'}, inplace=True) # Rename title to title2 to avoid name clash with title column from DataFrame title1
        link_type.rename(columns={'id': 'link_type_id'}, inplace=True)
        keyword.rename(columns={'id': 'keyword_id'}, inplace=True)

        return [keyword, link_type, movie_keyword, movie_link, title1, title2]

    def postprocess(self, result: pd.DataFrame) -> pd.DataFrame:
        # Project result
        result.drop(columns=list(result.columns.difference(['link', 'title', 'title2'])), inplace=True)

        if self.is_aggregation_enabled():
            # Apply aggregation
            logging.debug('Applying aggregation...')
            result = pd.DataFrame({'link_type': [result['link'].min()], 'first_movie': [result['title'].min()], 'second_movie': [result['title2'].min()]})
        else:
            # Just perform renaming
            result.rename(columns={'link': 'link_type', 'title': 'first_movie', 'title2': 'second_movie'}, inplace=True)
        
        return result

# LUBM: Q1
# out (x) : - takesCourse (x, 'http://www.Department0.University0.edu/GraduateCourse0'), type (x, 'GraduateStudent').
class LUBM_Q1(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q1', '../sql_queries/lubm/q1.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        takesCourse = self.dataset.load_takesCourse()
        type = self.dataset.load_type()

        # Apply filter
        logging.debug('Applying filter...')
        takesCourse = takesCourse[takesCourse['object'] == 'http://www.Department0.University0.edu/GraduateCourse0']
        type = type[type['object'] == 'GraduateStudent']

        # For each DataFrame, drop the column for which a literal is used in the query
        takesCourse = takesCourse.drop(columns=['object'])
        type = type.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        takesCourse.columns = ['x']
        type.columns = ['x']

        return [takesCourse, type]

# LUBM: Q2
# out (x,y,z) : - memberOf (x, y), subOrganizationOf (y, z), undergraduateDegreeFrom (x,z), type (x, 'GraduateStudent'), type (y, 'Department'), type (z, 'University').
class LUBM_Q2(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q2', '../sql_queries/lubm/q2.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        memberOf = self.dataset.load_memberOf()
        subOrganizationOf = self.dataset.load_subOrganizationOf()
        undergraduateDegreeFrom = self.dataset.load_undergraduateDegreeFrom()
        type = self.dataset.load_type()
        type2 = type.copy(deep=True)
        type3 = type.copy(deep=True)

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'GraduateStudent']
        type2 = type2[type2['object'] == 'Department']
        type3 = type3[type3['object'] == 'University']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        type2 = type2.drop(columns=['object'])
        type3 = type3.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        memberOf.columns = ['x', 'y']
        subOrganizationOf.columns = ['y', 'z']
        undergraduateDegreeFrom.columns = ['x', 'z']
        type.columns = ['x']
        type2.columns = ['y']
        type3.columns = ['z']

        return [memberOf, subOrganizationOf, undergraduateDegreeFrom, type, type2, type3]

# LUBM: Q3
# out (x) : - type (x, 'Publication'), publicationAuthor (x, 'http://www.Department0.University0.edu/AssistantProfessor0').
class LUBM_Q3(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q3', '../sql_queries/lubm/q3.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()
        publicationAuthor = self.dataset.load_publicationAuthor()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'Publication']
        publicationAuthor = publicationAuthor[publicationAuthor['object'] == 'http://www.Department0.University0.edu/AssistantProfessor0']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        publicationAuthor = publicationAuthor.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        type.columns = ['x']
        publicationAuthor.columns = ['x']

        return [type, publicationAuthor]
    
# LUBM: Q4
# out (x, y, z, w) : - worksFor (x, 'http://www.Department0.University0.edu'), name (x, y), emailAddress (x, w), telephone (x, z), type (x, 'AssociateProfessor').
class LUBM_Q4(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q4', '../sql_queries/lubm/q4.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        worksFor = self.dataset.load_worksFor()
        name = self.dataset.load_name()
        emailAddress = self.dataset.load_emailAddress()
        telephone = self.dataset.load_telephone()
        type = self.dataset.load_type()

        # Apply filter
        logging.debug('Applying filter...')
        worksFor = worksFor[worksFor['object'] == 'http://www.Department0.University0.edu']
        type = type[type['object'] == 'AssociateProfessor']

        # For each DataFrame, drop the column for which a literal is used in the query
        worksFor = worksFor.drop(columns=['object'])
        type = type.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        worksFor.columns = ['x']
        name.columns = ['x', 'y']
        emailAddress.columns = ['x', 'w']
        telephone.columns = ['x', 'z']
        type.columns = ['x']

        return [worksFor, name, emailAddress, telephone, type]

# LUBM: Q5
# out (x) : - type (x, 'UndergraduateStudent'), memberOf (x, 'http://www.Department0.University0.edu').
class LUBM_Q5(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q5', '../sql_queries/lubm/q5.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()
        memberOf = self.dataset.load_memberOf()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'UndergraduateStudent']
        memberOf = memberOf[memberOf['object'] == 'http://www.Department0.University0.edu']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        memberOf = memberOf.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        type.columns = ['x']
        memberOf.columns = ['x']

        return [type, memberOf]

# LUBM: Q7
# out (x,y) : - teacherOf ('http://www.Department0.University0.edu/AssociateProfessor0',x), takesCourse (y, x), type (x, 'Course'), type (y, 'UndergraduateStudent').
class LUBM_Q7(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q7', '../sql_queries/lubm/q7.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        teacherOf = self.dataset.load_teacherOf()
        takesCourse = self.dataset.load_takesCourse()
        type = self.dataset.load_type()
        type2 = type.copy(deep=True)

        # Apply filter
        logging.debug('Applying filter...')
        teacherOf = teacherOf[teacherOf['subject'] == 'http://www.Department0.University0.edu/AssociateProfessor0']
        type = type[type['object'] == 'Course']
        type2 = type2[type2['object'] == 'UndergraduateStudent']

        # For each DataFrame, drop the column for which a literal is used in the query
        teacherOf = teacherOf.drop(columns=['subject'])
        type = type.drop(columns=['object'])
        type2 = type2.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        teacherOf.columns = ['x']
        takesCourse.columns = ['y', 'x']
        type.columns = ['x']
        type2.columns = ['y']

        return [teacherOf, takesCourse, type, type2]

# LUBM: Q8
# out (x,y,z) : - memberOf (x, y), emailAddress (x, z), type (x, 'UndergraduateStudent'), subOrganizationOf(y, 'http://www.University0.edu'), type (y, 'Department').
class LUBM_Q8(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q8', '../sql_queries/lubm/q8.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        memberOf = self.dataset.load_memberOf()
        emailAddress = self.dataset.load_emailAddress()
        type = self.dataset.load_type()
        subOrganizationOf = self.dataset.load_subOrganizationOf()
        type2 = type.copy(deep=True)

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'UndergraduateStudent']
        subOrganizationOf = subOrganizationOf[subOrganizationOf['object'] == 'http://www.University0.edu']
        type2 = type2[type2['object'] == 'Department']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        subOrganizationOf = subOrganizationOf.drop(columns=['object'])
        type2 = type2.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        memberOf.columns = ['x', 'y']
        emailAddress.columns = ['x', 'z']
        type.columns = ['x']
        subOrganizationOf.columns = ['y']
        type2.columns = ['y']

        return [memberOf, emailAddress, type, subOrganizationOf, type2]

# LUBM: Q9
# out (x,y,z) : - type (x, 'UndergraduateStudent'), type (y, 'Course'), type (z, 'AssistantProfessor'), advisor (x,z), teacherOf (z,y), takesCourse (x, y) .
class LUBM_Q9(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q9', '../sql_queries/lubm/q9.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()
        type2 = type.copy(deep=True)
        type3 = type.copy(deep=True)
        advisor = self.dataset.load_advisor()
        teacherOf = self.dataset.load_teacherOf()
        takesCourse = self.dataset.load_takesCourse()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'UndergraduateStudent']
        type2 = type2[type2['object'] == 'Course']
        type3 = type3[type3['object'] == 'AssistantProfessor']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        type2 = type2.drop(columns=['object'])
        type3 = type3.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrames
        advisor.columns = ['x', 'z']
        teacherOf.columns = ['z', 'y']
        takesCourse.columns = ['x', 'y']
        type.columns = ['x']
        type2.columns = ['y']
        type3.columns = ['z']

        return [type, type2, type3, advisor, teacherOf, takesCourse]

# LUBM: Q11
# out (x) : - type (x, 'ResearchGroup'), subOrganizationOf(x,'http://www.University0.edu').
class LUBM_Q11(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q11', '../sql_queries/lubm/q11.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()
        subOrganizationOf = self.dataset.load_subOrganizationOf()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'ResearchGroup']
        subOrganizationOf = subOrganizationOf[subOrganizationOf['object'] == 'http://www.University0.edu']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        subOrganizationOf = subOrganizationOf.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        type.columns = ['x']
        subOrganizationOf.columns = ['x']

        return [type, subOrganizationOf]

# LUBM: Q12
# out (x,y) : - worksFor (y, x), type (y, 'FullProfessor'), subOrganizationOf (x, 'http://www.University0.edu'), type (x, 'Department').
class LUBM_Q12(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q12', '../sql_queries/lubm/q12.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        worksFor = self.dataset.load_worksFor()
        type = self.dataset.load_type()
        subOrganizationOf = self.dataset.load_subOrganizationOf()
        type2 = type.copy(deep=True)

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'FullProfessor']
        subOrganizationOf = subOrganizationOf[subOrganizationOf['object'] == 'http://www.University0.edu']
        type2 = type2[type2['object'] == 'Department']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        subOrganizationOf = subOrganizationOf.drop(columns=['object'])
        type2 = type2.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        worksFor.columns = ['y', 'x']
        type.columns = ['y']
        subOrganizationOf.columns = ['x']
        type2.columns = ['x']

        return [worksFor, type, subOrganizationOf, type2]

# LUBM: Q13
# out (x) : - type (x, 'GraduateStudent'), undergraduateDegreeFrom(x, 'http://www.University567.edu').
class LUBM_Q13(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q13', '../sql_queries/lubm/q13.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()
        undergraduateDegreeFrom = self.dataset.load_undergraduateDegreeFrom()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'GraduateStudent']
        undergraduateDegreeFrom = undergraduateDegreeFrom[undergraduateDegreeFrom['object'] == 'http://www.University567.edu']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])
        undergraduateDegreeFrom = undergraduateDegreeFrom.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        type.columns = ['x']
        undergraduateDegreeFrom.columns = ['x']

        return [type, undergraduateDegreeFrom]

# LUBM: Q14
# out (x) : - type (x, 'UndergraduateStudent').
class LUBM_Q14(JoinQuery):
    def __init__(self, dataset: LUBMDataset):
        super().__init__(dataset, 'Q14', '../sql_queries/lubm/q14.sql')
    
    def preprocess(self) -> list[pd.DataFrame]:
        # Load data
        logging.debug('Loading data...')
        type = self.dataset.load_type()

        # Apply filter
        logging.debug('Applying filter...')
        type = type[type['object'] == 'UndergraduateStudent']

        # For each DataFrame, drop the column for which a literal is used in the query
        type = type.drop(columns=['object'])

        # Rename columns depending on the joins that will be performed
        # Let the second column name be a random string for the type DataFrame
        type.columns = ['x']

        return [type]

# out(X, Y, Z) :- edge(X, Y), edge(Y, Z), edge(Z, X).
class TriangleJoinQuery(JoinQuery):
    def preprocess(self) -> list[pd.DataFrame]:
        df = self.dataset.load()

        # To perform the join, we need three DataFrames, i.e. we need two additional copies of the DataFrame.
        df2 = df.copy(deep=True)
        df3 = df.copy(deep=True)

        # Rename the columns of the DataFrames to match the join conditions.
        df.columns = ['x', 'y']
        df2.columns = ['y', 'z']
        df3.columns = ['z', 'x']

        return [df, df2, df3]

class Facebook_Query(TriangleJoinQuery):
    def __init__(self, dataset: FacebookDataset):
        super().__init__(dataset, 'Facebook_Query', '../sql_queries/triangle/facebook/query.sql')
    
class Arxiv_Query(TriangleJoinQuery):
    def __init__(self, dataset: ArxivDataset):
        super().__init__(dataset, 'Arxiv_Query', '../sql_queries/triangle/arxiv/query.sql')

class GPlus_Query(TriangleJoinQuery):
    def __init__(self, dataset: GPlusDataset):
        super().__init__(dataset, 'GPlus_Query', '../sql_queries/triangle/gplus/query.sql')
    
class LiveJournal_Query(TriangleJoinQuery):
    def __init__(self, dataset: LiveJournalDataset):
        super().__init__(dataset, 'LiveJournal_Query', '../sql_queries/triangle/livejournal/query.sql')
    
class Orkut_Query(TriangleJoinQuery):
    def __init__(self, dataset: OrkutDataset):
        super().__init__(dataset, 'Orkut_Query', '../sql_queries/triangle/orkut/query.sql')

class Patents_Query(TriangleJoinQuery):
    def __init__(self, dataset: PatentsDataset):
        super().__init__(dataset, 'Patents_Query', '../sql_queries/triangle/patents/query.sql')

def get_all_queries() -> dict[str, dict[str, type[JoinQuery]]]:
    return {
        'LUBM': {
            'Q1': LUBM_Q1,
            'Q2': LUBM_Q2,
            'Q3': LUBM_Q3,
            'Q4': LUBM_Q4,
            'Q5': LUBM_Q5,
            'Q7': LUBM_Q7,
            'Q8': LUBM_Q8,
            #'Q9': LUBM_Q9, # memory overflow
            'Q11': LUBM_Q11,
            'Q12': LUBM_Q12,
            'Q13': LUBM_Q13,
            'Q14': LUBM_Q14
        },
        'IMDB': {
            '2a': IMDB_2a,
            '2b': IMDB_2b,
            '2c': IMDB_2c,
            '2d': IMDB_2d,
            '4c': IMDB_4c,
            '8c': IMDB_8c,
            '8d': IMDB_8d,
            '10b': IMDB_10b,
            '10c': IMDB_10c,
            '13a': IMDB_13a,
            '13d': IMDB_13d,
            '16b': IMDB_16b,
            '17e': IMDB_17e,
            '32a': IMDB_32a
        },
        'Facebook': {
            'Facebook_Query': Facebook_Query
        },
        'Arxiv': {
            'Arxiv_Query': Arxiv_Query
        },
        'GPlus': {
            #'GPlus_Query': GPlus_Query # memory overflow
        },
        'LiveJournal': {
            #'LiveJournal_Query': LiveJournal_Query # memory overflow
        },
        'Orkut': {
            #'Orkut_Query': Orkut_Query # memory overflow
        },
        'Patents': {
            #'Patents_Query': Patents_Query # memory overflow
        }
    }