#!/usr/bin/env python
# coding: utf-8

# CALL compute_df_score like this
#compute_df_score(2, 4, ['Action', 'Comedy'], ['20th Century Fox'], ['Christopher Nolan'],
#                      ['Christopher Nolan'], ['John Belushi'], ['Fred Astaire'])


import pandas as pd
import csv

name_basics = pd.read_csv('cleaned_data/relevant_name_basics.txt', delimiter='|')

def load_dict(dict_path):
    with open(dict_path) as csv_file:
        reader = csv.reader(csv_file)
        return dict(reader)

nconst_to_vote_average_score = load_dict('cleaned_data/nconst_to_vote_average_score.csv')
nconst_to_vote_count_score = load_dict('cleaned_data/nconst_to_vote_count_score.csv')
nconst_to_profit_score = load_dict('cleaned_data/nconst_to_profit_score.csv')
nconst_to_rev_budget_ratio_score = load_dict('cleaned_data/nconst_to_rev_budget_ratio_score.csv')

nontrivial_col_types = [['prod_company_1',
  'prod_company_2',
  'prod_company_3',
  'prod_company_4',
  'prod_company_5'],
 ['writer_1', 'writer_2', 'writer_3', 'writer_4', 'writer_5'],
 ['director_1', 'director_2', 'director_3'],
 ['cast_1',
  'cast_2',
  'cast_3',
  'cast_4',
  'cast_5',
  'cast_6',
  'cast_7',
  'cast_8',
  'cast_9',
  'cast_10'],
 ['others_1',
  'others_2',
  'others_3',
  'others_4',
  'others_5',
  'others_6',
  'others_7',
  'others_8',
  'others_9',
  'others_10']]

num_genres = 4
num_prod_companies = 5
num_directors = 3
num_writers = 5
num_others = 10
num_cast = 10

def get_nconst(name):
    try:
        return name_basics[name_basics['primaryName'] == name][['nconst']].values[0][0]
    except:
        return ''
    
def set_nconsts(staff_list):
    for i in range(len(staff_list)):
        staff_list[i] = get_nconst(staff_list[i])
        
    return staff_list

def fill_missing(feature_list, max_number):
    print(feature_list)
    i = len(feature_list)
    
    while i < max_number:
        feature_list.append('')
        i += 1
        
    return feature_list

def assign_features(cur_dict, col, col_values, num_cols):
    for i in range(num_cols):
        cur_dict[col + '_' + str(i + 1)] = col_values[i]
    return cur_dict

    
def get_df_without_scores(budget, runtime, genres, prod_companies, writers, directors, actors, others):
    """
    prod_companies, writers, directors, actors, others will be a list of names which I'll then join on their id's
    which I'll then use to compute the score
    """
    genres = fill_missing(genres, num_genres)
    print(prod_companies)
    prod_companies = fill_missing(prod_companies, num_prod_companies)
    writers = fill_missing(set_nconsts(writers), num_writers)
    directors = fill_missing(set_nconsts(directors), num_directors)
    actors = fill_missing(set_nconsts(actors), num_cast)
    others = fill_missing(set_nconsts(others), num_others)
        
    df_dict = {}
    
    df_dict['budget'] = budget
    df_dict['runtimeMinutes'] = runtime
    
    df_dict = assign_features(df_dict, 'genre', genres, num_genres)
    df_dict = assign_features(df_dict, 'prod_company', prod_companies, num_prod_companies)
    df_dict = assign_features(df_dict, 'writer', writers, num_writers)
    df_dict = assign_features(df_dict, 'director', directors, num_directors)
    df_dict = assign_features(df_dict, 'cast', actors, num_cast)
    df_dict = assign_features(df_dict, 'others', others, num_others)

    return pd.DataFrame([df_dict])

def compute_df_score(budget, runtime, genres, prod_companies, writers, directors, actors, others):
    df = get_df_without_scores(budget, runtime, genres, prod_companies,
                                              writers, directors, actors, others)
    
    for index, row in df.iterrows():
        for col_type in nontrivial_col_types:
            for col in col_type:
                id_ = row[col]
                # if id_ != '': 
                try:
                    df.set_value(index, col + '_' + 'vote_average',
                                                nconst_to_vote_average_score[id_])
                    df.set_value(index, col + '_' + 'vote_count',
                                            nconst_to_vote_count_score[id_])
                    df.set_value(index, col + '_' + 'profit',
                                                nconst_to_profit_score[id_])
                    df.set_value(index, col + '_' + 'rev_budget_ratio',
                                                nconst_to_rev_budget_ratio_score[id_])
                except:
                    df.set_value(index, col + '_' + 'vote_average', 0)
                    df.set_value(index, col + '_' + 'vote_count', 0)
                    df.set_value(index, col + '_' + 'profit', 0)
                    df.set_value(index, col + '_' + 'rev_budget_ratio', 0)
    
    for col_type in nontrivial_col_types:
        for col in col_type:
            df = df.drop(col, axis=1)
    
    return df


# In[2]:


#compute_df_score(2, 4, ['Action', 'Comedy'], ['20th Century Fox'], ['Christopher Nolan'],
#                      ['Christopher Nolan'], ['John Belushi'], ['Fred Astaire'])


# In[ ]:




