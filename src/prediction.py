import pickle
import numpy as np
from component_score_rt import compute_df_score

def ml_prediction(df,rev_model,rating_model):

    # Pre Computed Constants from Training Data
    budget_mean = 45129357.13983431
    budget_std  = 49040822.82769277
    runtime_max = 260

    log_rev_budget_ratio_mean = 0.604557343641753 
    log_rev_budget_ratio_std  = 1.7679498325555352
    
    budget = df['budget'][0]

    # One Hot the Genre
    genres = ['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','War','Western']
    for g in genres:
        df[g] = ((df['genre_1'].str.contains(g))| (df['genre_2'].str.contains(g))| (df['genre_3'].str.contains(g))| (df['genre_4'].str.contains(g))).astype(float)

    # Preparing data for ML model
    df['budget']         = np.log(df['budget'])
    df['budget']         = (df['budget']-budget_mean)/budget_std
    df['runtimeMinutes'] = df['runtimeMinutes']/runtime_max
    input_features = df[['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','War','Western','budget','runtimeMinutes','prod_company_1_vote_average','prod_company_2_vote_average','prod_company_3_vote_average','prod_company_4_vote_average','prod_company_5_vote_average','writer_1_vote_average','writer_2_vote_average','writer_3_vote_average','writer_4_vote_average','writer_5_vote_average','director_1_vote_average','director_2_vote_average','director_3_vote_average','cast_1_vote_average','cast_2_vote_average','cast_3_vote_average','cast_4_vote_average','cast_5_vote_average','cast_6_vote_average','cast_7_vote_average','cast_8_vote_average','cast_9_vote_average','cast_10_vote_average','others_1_vote_average','others_2_vote_average','others_3_vote_average','others_4_vote_average','others_5_vote_average','others_6_vote_average','others_7_vote_average','others_8_vote_average','others_9_vote_average','others_10_vote_average','prod_company_1_vote_count','prod_company_2_vote_count','prod_company_3_vote_count','prod_company_4_vote_count','prod_company_5_vote_count','writer_1_vote_count','writer_2_vote_count','writer_3_vote_count','writer_4_vote_count','writer_5_vote_count','director_1_vote_count','director_2_vote_count','director_3_vote_count','cast_1_vote_count','cast_2_vote_count','cast_3_vote_count','cast_4_vote_count','cast_5_vote_count','cast_6_vote_count','cast_7_vote_count','cast_8_vote_count','cast_9_vote_count','cast_10_vote_count','others_1_vote_count','others_2_vote_count','others_3_vote_count','others_4_vote_count','others_5_vote_count','others_6_vote_count','others_7_vote_count','others_8_vote_count','others_9_vote_count','others_10_vote_count','prod_company_1_profit','prod_company_2_profit','prod_company_3_profit','prod_company_4_profit','prod_company_5_profit','writer_1_profit','writer_2_profit','writer_3_profit','writer_4_profit','writer_5_profit','director_1_profit','director_2_profit','director_3_profit','cast_1_profit','cast_2_profit','cast_3_profit','cast_4_profit','cast_5_profit','cast_6_profit','cast_7_profit','cast_8_profit','cast_9_profit','cast_10_profit','others_1_profit','others_2_profit','others_3_profit','others_4_profit','others_5_profit','others_6_profit','others_7_profit','others_8_profit','others_9_profit','others_10_profit','prod_company_1_rev_budget_ratio','prod_company_2_rev_budget_ratio','prod_company_3_rev_budget_ratio','prod_company_4_rev_budget_ratio','prod_company_5_rev_budget_ratio','writer_1_rev_budget_ratio','writer_2_rev_budget_ratio','writer_3_rev_budget_ratio','writer_4_rev_budget_ratio','writer_5_rev_budget_ratio','director_1_rev_budget_ratio','director_2_rev_budget_ratio','director_3_rev_budget_ratio','cast_1_rev_budget_ratio','cast_2_rev_budget_ratio','cast_3_rev_budget_ratio','cast_4_rev_budget_ratio','cast_5_rev_budget_ratio','cast_6_rev_budget_ratio','cast_7_rev_budget_ratio','cast_8_rev_budget_ratio','cast_9_rev_budget_ratio','cast_10_rev_budget_ratio','others_1_rev_budget_ratio','others_2_rev_budget_ratio','others_3_rev_budget_ratio','others_4_rev_budget_ratio','others_5_rev_budget_ratio','others_6_rev_budget_ratio','others_7_rev_budget_ratio','others_8_rev_budget_ratio','others_9_rev_budget_ratio','others_10_rev_budget_ratio']]
    
    pred_log_rev_budget_ratio = rev_model.predict(input_features)[0]
    pred_rating_0_to_1        = rating_model.predict(input_features)[0]

    pred_rev  = np.exp(((pred_log_rev_budget_ratio*log_rev_budget_ratio_std)+log_rev_budget_ratio_mean))*budget
    pred_rate = pred_rating_0_to_1*5

    return pred_rev,pred_rate

def predict(
        genres=None,
        budget=1000000,
        runtime=90,
        directors=None,
        writers=None,
        cast=None,
        prod_companies=None
):


    rev_model = pickle.load(open('ML/models/revenue_model_rf.p', 'rb'))
    rating_model = pickle.load(open('ML/models/rating_model_rf.p', 'rb'))

    x = compute_df_score(budget, runtime, genres, prod_companies, writers,
                         directors, cast, [])
    rev,rating = ml_prediction(x,rev_model,rating_model)

    return {
        "Revenue":rev,
        "Rating": rating
    }












