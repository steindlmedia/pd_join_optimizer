import pandas as pd
import os
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, base_path: str):
        self.path = os.path.join(base_path, self.get_schema_name())

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_schema_name(self):
        pass

class IMDBDataset(Dataset):
    def get_name(self):
        return 'IMDB'
    
    def get_schema_name(self):
        return 'imdb'
    
    # CREATE TABLE title (id integer NOT NULL PRIMARY KEY, title character varying NOT NULL, imdb_index character varying(5), kind_id integer NOT NULL, production_year integer, imdb_id integer, phonetic_code character varying(5), episode_of_id integer, season_nr integer, episode_nr integer, series_years character varying(49), md5sum character varying(32));
    def load_title(self):
        return pd.read_csv(os.path.join(self.path, 'title.csv'), sep=',', header=None, names=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], dtype={'id': 'Int64', 'imdb_index': 'string', 'kind_id': 'Int64', 'production_year': 'Int64', 'imdb_id': 'Int64', 'phonetic_code': 'string', 'episode_of_id': 'Int64', 'season_nr': 'Int64', 'episode_nr': 'Int64', 'series_years': 'string', 'md5sum': 'string'}, keep_default_na=False)
    
    # CREATE TABLE movie_keyword (id integer NOT NULL PRIMARY KEY, movie_id integer NOT NULL, keyword_id integer NOT NULL);
    def load_movie_keyword(self):
        return pd.read_csv(os.path.join(self.path, 'movie_keyword.csv'), sep=',', header=None, names=['id', 'movie_id', 'keyword_id'], dtype={'id': 'Int64', 'movie_id': 'Int64', 'keyword_id': 'Int64'}, keep_default_na=False)

    # CREATE TABLE movie_companies (id integer NOT NULL PRIMARY KEY, movie_id integer NOT NULL, company_id integer NOT NULL, company_type_id integer NOT NULL, note character varying);
    def load_movie_companies(self):
        return pd.read_csv(os.path.join(self.path, 'movie_companies.csv'), sep=',', header=None, names=['id', 'movie_id', 'company_id', 'company_type_id', 'note'], dtype={'id': 'Int64', 'movie_id': 'Int64', 'company_id': 'Int64', 'company_type_id': 'Int64', 'note': 'string'}, keep_default_na=False)

    # CREATE TABLE keyword (id integer NOT NULL PRIMARY KEY, keyword character varying NOT NULL, phonetic_code character varying(5));
    def load_keyword(self):
        return pd.read_csv(os.path.join(self.path, 'keyword.csv'), sep=',', header=None, names=['id', 'keyword', 'phonetic_code'], dtype={'id': 'Int64', 'keyword': 'string', 'phonetic_code': 'string'}, keep_default_na=False)

    # CREATE TABLE company_name (id integer NOT NULL PRIMARY KEY, name character varying NOT NULL, country_code character varying(6), imdb_id integer, name_pcode_nf character varying(5), name_pcode_sf character varying(5), md5sum character varying(32));
    def load_company_name(self):
        return pd.read_csv(os.path.join(self.path, 'company_name.csv'), sep=',', header=None, names=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], dtype={'id': 'Int64', 'name': 'string', 'country_code': 'string', 'imdb_id': 'Int64', 'name_pcode_nf': 'string', 'name_pcode_sf': 'string', 'md5sum': 'string'}, keep_default_na=False)

    # CREATE TABLE movie_info_idx (id integer NOT NULL PRIMARY KEY, movie_id integer NOT NULL, info_type_id integer NOT NULL, info character varying NOT NULL, note character varying(1));
    def load_movie_info_idx(self):
        return pd.read_csv(os.path.join(self.path, 'movie_info_idx.csv'), sep=',', header=None, names=['id', 'movie_id', 'info_type_id', 'info', 'note'], dtype={'id': 'Int64', 'movie_id': 'Int64', 'info_type_id': 'Int64', 'info': 'string', 'note': 'string'}, keep_default_na=False)

    # CREATE TABLE info_type (id integer NOT NULL PRIMARY KEY, info character varying(32) NOT NULL);
    def load_info_type(self):
        return pd.read_csv(os.path.join(self.path, 'info_type.csv'), sep=',', header=None, names=['id', 'info'], dtype={'id': 'Int64', 'info': 'string'}, keep_default_na=False)

    # CREATE TABLE role_type (id integer NOT NULL PRIMARY KEY, role character varying(32) NOT NULL);
    def load_role_type(self):
        return pd.read_csv(os.path.join(self.path, 'role_type.csv'), sep=',', header=None, names=['id', 'role'], dtype={'id': 'Int64', 'role': 'string'}, keep_default_na=False)

    # CREATE TABLE cast_info (id integer NOT NULL PRIMARY KEY, person_id integer NOT NULL, movie_id integer NOT NULL, person_role_id integer, note character varying, nr_order integer, role_id integer NOT NULL);
    def load_cast_info(self):
        return pd.read_csv(os.path.join(self.path, 'cast_info.csv'), sep=',', header=None, names=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'], dtype={'id': 'Int64', 'person_id': 'Int64', 'movie_id': 'Int64', 'person_role_id': 'Int64', 'note': 'string', 'nr_order': 'Int64', 'role_id': 'Int64'}, keep_default_na=False)

    # CREATE TABLE aka_name (id integer NOT NULL PRIMARY KEY, person_id integer NOT NULL, name character varying, imdb_index character varying(3), name_pcode_cf character varying(11), name_pcode_nf character varying(11), surname_pcode character varying(11), md5sum character varying(65));
    def load_aka_name(self):
        return pd.read_csv(os.path.join(self.path, 'aka_name.csv'), sep=',', header=None, names=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], dtype={'id': 'Int64', 'person_id': 'Int64', 'name': 'string', 'imdb_index': 'string', 'name_pcode_cf': 'string', 'name_pcode_nf': 'string', 'surname_pcode': 'string', 'md5sum': 'string'}, keep_default_na=False)

    # CREATE TABLE company_type (id integer NOT NULL PRIMARY KEY, kind character varying(32));
    def load_company_type(self):
        return pd.read_csv(os.path.join(self.path, 'company_type.csv'), sep=',', header=None, names=['id', 'kind'], dtype={'id': 'Int64', 'kind': 'string'}, keep_default_na=False)

    # CREATE TABLE char_name (id integer NOT NULL PRIMARY KEY, name character varying NOT NULL, imdb_index character varying(2), imdb_id integer, name_pcode_nf character varying(5), surname_pcode character varying(5), md5sum character varying(32));
    def load_char_name(self):
        return pd.read_csv(os.path.join(self.path, 'char_name.csv'), sep=',', header=None, names=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'], dtype={'id': 'Int64', 'name': 'string', 'imdb_index': 'string', 'imdb_id': 'Int64', 'name_pcode_nf': 'string', 'surname_pcode': 'string', 'md5sum': 'string'}, keep_default_na=False)

    # CREATE TABLE movie_info (id integer NOT NULL PRIMARY KEY, movie_id integer NOT NULL, info_type_id integer NOT NULL, info character varying NOT NULL, note character varying);
    def load_movie_info(self):
        return pd.read_csv(os.path.join(self.path, 'movie_info.csv'), sep=',', header=None, names=['id', 'movie_id', 'info_type_id', 'info', 'note'], dtype={'id': 'Int64', 'movie_id': 'Int64', 'info_type_id': 'Int64', 'info': 'string', 'note': 'string'}, keep_default_na=False)

    # CREATE TABLE kind_type (id integer NOT NULL PRIMARY KEY, kind character varying(15));
    def load_kind_type(self):
        return pd.read_csv(os.path.join(self.path, 'kind_type.csv'), sep=',', header=None, names=['id', 'kind'], dtype={'id': 'Int64', 'kind': 'string'}, keep_default_na=False)

    # CREATE TABLE name (id integer NOT NULL PRIMARY KEY, name character varying NOT NULL, imdb_index character varying(9), imdb_id integer, gender character varying(1), name_pcode_cf character varying(5), name_pcode_nf character varying(5), surname_pcode character varying(5), md5sum character varying(32));
    def load_name(self):
        return pd.read_csv(os.path.join(self.path, 'name.csv'), sep=',', header=None, names=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], dtype={'id': 'Int64', 'name': 'string', 'imdb_index': 'string', 'imdb_id': 'Int64', 'gender': 'string', 'name_pcode_cf': 'string', 'name_pcode_nf': 'string', 'surname_pcode': 'string', 'md5sum': 'string'}, keep_default_na=False)

    # CREATE TABLE movie_link (id integer NOT NULL PRIMARY KEY, movie_id integer NOT NULL, linked_movie_id integer NOT NULL, link_type_id integer NOT NULL);
    def load_movie_link(self):
        return pd.read_csv(os.path.join(self.path, 'movie_link.csv'), sep=',', header=None, names=['id', 'movie_id', 'linked_movie_id', 'link_type_id'], dtype={'id': 'Int64', 'movie_id': 'Int64', 'linked_movie_id': 'Int64', 'link_type_id': 'Int64'}, keep_default_na=False)

    # CREATE TABLE link_type (id integer NOT NULL PRIMARY KEY, link character varying(32) NOT NULL);
    def load_link_type(self):
        return pd.read_csv(os.path.join(self.path, 'link_type.csv'), sep=',', header=None, names=['id', 'link'], dtype={'id': 'Int64', 'link': 'string'}, keep_default_na=False)

class LUBMDataset(Dataset):
    def get_name(self):
        return 'LUBM'
    
    def get_schema_name(self):
        return 'lubm'

    def load_takesCourse(self):
        return pd.read_csv(os.path.join(self.path, 'takesCourse.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)
    
    def load_undergraduateDegreeFrom(self):
        return pd.read_csv(os.path.join(self.path, 'undergraduateDegreeFrom.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_subOrganizationOf(self):
        return pd.read_csv(os.path.join(self.path, 'subOrganizationOf.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_memberOf(self):
        return pd.read_csv(os.path.join(self.path, 'memberOf.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_publicationAuthor(self):
        return pd.read_csv(os.path.join(self.path, 'publicationAuthor.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_type(self):
        return pd.read_csv(os.path.join(self.path, 'type.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_telephone(self):
        return pd.read_csv(os.path.join(self.path, 'telephone.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_emailAddress(self):
        return pd.read_csv(os.path.join(self.path, 'emailAddress.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_name(self):
        return pd.read_csv(os.path.join(self.path, 'name.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_worksFor(self):
        return pd.read_csv(os.path.join(self.path, 'worksFor.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_teacherOf(self):
        return pd.read_csv(os.path.join(self.path, 'teacherOf.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

    def load_advisor(self):
        return pd.read_csv(os.path.join(self.path, 'advisor.csv'), sep=',', header=None, names=['subject', 'object'], dtype={'subject': 'string', 'object': 'string'}, keep_default_na=False)

class FacebookDataset(Dataset):
    def get_name(self):
        return 'Facebook'
    
    def get_schema_name(self):
        return 'facebook'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "facebook_combined.txt"), sep=" ", skiprows=0, names=["x", "y"], dtype={'x': 'Int64', 'y': 'Int64'}, keep_default_na=False)

class ArxivDataset(Dataset):
    def get_name(self):
        return 'Arxiv'
    
    def get_schema_name(self):
        return 'arxiv'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "ca-GrQc.txt"), sep="\t", names=["x", "y"], dtype={'x': 'Int64', 'y': 'Int64'}, keep_default_na=False)

class GPlusDataset(Dataset):
    def get_name(self):
        return 'G+'
    
    def get_schema_name(self):
        return 'gplus'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "gplus_combined.txt"), sep=" ", names=["x", "y"], dtype={'x': 'string', 'y': 'string'}, keep_default_na=False)

class LiveJournalDataset(Dataset):
    def get_name(self):
        return 'LiveJournal'
    
    def get_schema_name(self):
        return 'livejournal'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "soc-LiveJournal1.txt"), sep="\t", names=["x", "y"], dtype={'x': 'Int64', 'y': 'Int64'}, keep_default_na=False)

class OrkutDataset(Dataset):
    def get_name(self):
        return 'Orkut'
    
    def get_schema_name(self):
        return 'orkut'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "com-orkut.ungraph.txt"), sep="\t", names=["x", "y"], dtype={'x': 'Int64', 'y': 'Int64'}, keep_default_na=False)

class PatentsDataset(Dataset):
    def get_name(self):
        return 'Patents'
    
    def get_schema_name(self):
        return 'patents'
    
    def load(self):
        return pd.read_csv(os.path.join(self.path, "cit-Patents.txt"), sep="\t", names=["x", "y"], dtype={'x': 'Int64', 'y': 'Int64'}, keep_default_na=False)
    
def get_all_datasets() -> dict[str, type[Dataset]]:
    return {
        'IMDB': IMDBDataset,
        'LUBM': LUBMDataset,
        'Facebook': FacebookDataset,
        'Arxiv': ArxivDataset,
        'GPlus': GPlusDataset,
        'LiveJournal': LiveJournalDataset,
        'Orkut': OrkutDataset,
        'Patents': PatentsDataset
    }