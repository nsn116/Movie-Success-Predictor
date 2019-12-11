import keras
import numpy as np
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Activation
from keras.activations import relu
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,BaggingRegressor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# For reproducibiity
np.random.seed(seed=2)

def preProcessData():
	'''
	More Data Preprocessing for ML
	'''
	# Grab Labels and Features + One hot genre
	df = pd.read_csv('final_data.csv')
	g1 =df['genre_1'].dropna().unique()
	g2 =df['genre_1'].dropna().unique()
	g3 =df['genre_1'].dropna().unique()
	g4 =df['genre_1'].dropna().unique()
	genres = np.unique(np.concatenate((g1,g2,g3,g4)))
	for g in genres:
		df[g] = ((df['genre_1'].str.contains(g))| (df['genre_2'].str.contains(g))| (df['genre_3'].str.contains(g))| (df['genre_4'].str.contains(g))).astype(float)

	# Normalize Vote Average to 0 - 1
	df['vote_average'] = df['vote_average']/10

	# Log some Numerical Values
	df['budget']		 = np.log(df['budget'])
	df['budget']  		 = (df['budget']-np.mean(df['budget']))/(np.std(df['budget']))
	df['runtimeMinutes'] = df['runtimeMinutes']/np.max(df['runtimeMinutes'])

	df['revenue']		   = np.log(df['revenue'])
	df['rev_budget_ratio'] = np.log(df['rev_budget_ratio'])

	labels = df[['vote_average','revenue','rev_budget_ratio']]
	feats = ['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','War','Western','budget','runtimeMinutes','prod_company_1_vote_average','prod_company_2_vote_average','prod_company_3_vote_average','prod_company_4_vote_average','prod_company_5_vote_average','writer_1_vote_average','writer_2_vote_average','writer_3_vote_average','writer_4_vote_average','writer_5_vote_average','director_1_vote_average','director_2_vote_average','director_3_vote_average','cast_1_vote_average','cast_2_vote_average','cast_3_vote_average','cast_4_vote_average','cast_5_vote_average','cast_6_vote_average','cast_7_vote_average','cast_8_vote_average','cast_9_vote_average','cast_10_vote_average','others_1_vote_average','others_2_vote_average','others_3_vote_average','others_4_vote_average','others_5_vote_average','others_6_vote_average','others_7_vote_average','others_8_vote_average','others_9_vote_average','others_10_vote_average','prod_company_1_vote_count','prod_company_2_vote_count','prod_company_3_vote_count','prod_company_4_vote_count','prod_company_5_vote_count','writer_1_vote_count','writer_2_vote_count','writer_3_vote_count','writer_4_vote_count','writer_5_vote_count','director_1_vote_count','director_2_vote_count','director_3_vote_count','cast_1_vote_count','cast_2_vote_count','cast_3_vote_count','cast_4_vote_count','cast_5_vote_count','cast_6_vote_count','cast_7_vote_count','cast_8_vote_count','cast_9_vote_count','cast_10_vote_count','others_1_vote_count','others_2_vote_count','others_3_vote_count','others_4_vote_count','others_5_vote_count','others_6_vote_count','others_7_vote_count','others_8_vote_count','others_9_vote_count','others_10_vote_count','prod_company_1_profit','prod_company_2_profit','prod_company_3_profit','prod_company_4_profit','prod_company_5_profit','writer_1_profit','writer_2_profit','writer_3_profit','writer_4_profit','writer_5_profit','director_1_profit','director_2_profit','director_3_profit','cast_1_profit','cast_2_profit','cast_3_profit','cast_4_profit','cast_5_profit','cast_6_profit','cast_7_profit','cast_8_profit','cast_9_profit','cast_10_profit','others_1_profit','others_2_profit','others_3_profit','others_4_profit','others_5_profit','others_6_profit','others_7_profit','others_8_profit','others_9_profit','others_10_profit','prod_company_1_rev_budget_ratio','prod_company_2_rev_budget_ratio','prod_company_3_rev_budget_ratio','prod_company_4_rev_budget_ratio','prod_company_5_rev_budget_ratio','writer_1_rev_budget_ratio','writer_2_rev_budget_ratio','writer_3_rev_budget_ratio','writer_4_rev_budget_ratio','writer_5_rev_budget_ratio','director_1_rev_budget_ratio','director_2_rev_budget_ratio','director_3_rev_budget_ratio','cast_1_rev_budget_ratio','cast_2_rev_budget_ratio','cast_3_rev_budget_ratio','cast_4_rev_budget_ratio','cast_5_rev_budget_ratio','cast_6_rev_budget_ratio','cast_7_rev_budget_ratio','cast_8_rev_budget_ratio','cast_9_rev_budget_ratio','cast_10_rev_budget_ratio','others_1_rev_budget_ratio','others_2_rev_budget_ratio','others_3_rev_budget_ratio','others_4_rev_budget_ratio','others_5_rev_budget_ratio','others_6_rev_budget_ratio','others_7_rev_budget_ratio','others_8_rev_budget_ratio','others_9_rev_budget_ratio','others_10_rev_budget_ratio']
	input_features = df[['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','War','Western','budget','runtimeMinutes','prod_company_1_vote_average','prod_company_2_vote_average','prod_company_3_vote_average','prod_company_4_vote_average','prod_company_5_vote_average','writer_1_vote_average','writer_2_vote_average','writer_3_vote_average','writer_4_vote_average','writer_5_vote_average','director_1_vote_average','director_2_vote_average','director_3_vote_average','cast_1_vote_average','cast_2_vote_average','cast_3_vote_average','cast_4_vote_average','cast_5_vote_average','cast_6_vote_average','cast_7_vote_average','cast_8_vote_average','cast_9_vote_average','cast_10_vote_average','others_1_vote_average','others_2_vote_average','others_3_vote_average','others_4_vote_average','others_5_vote_average','others_6_vote_average','others_7_vote_average','others_8_vote_average','others_9_vote_average','others_10_vote_average','prod_company_1_vote_count','prod_company_2_vote_count','prod_company_3_vote_count','prod_company_4_vote_count','prod_company_5_vote_count','writer_1_vote_count','writer_2_vote_count','writer_3_vote_count','writer_4_vote_count','writer_5_vote_count','director_1_vote_count','director_2_vote_count','director_3_vote_count','cast_1_vote_count','cast_2_vote_count','cast_3_vote_count','cast_4_vote_count','cast_5_vote_count','cast_6_vote_count','cast_7_vote_count','cast_8_vote_count','cast_9_vote_count','cast_10_vote_count','others_1_vote_count','others_2_vote_count','others_3_vote_count','others_4_vote_count','others_5_vote_count','others_6_vote_count','others_7_vote_count','others_8_vote_count','others_9_vote_count','others_10_vote_count','prod_company_1_profit','prod_company_2_profit','prod_company_3_profit','prod_company_4_profit','prod_company_5_profit','writer_1_profit','writer_2_profit','writer_3_profit','writer_4_profit','writer_5_profit','director_1_profit','director_2_profit','director_3_profit','cast_1_profit','cast_2_profit','cast_3_profit','cast_4_profit','cast_5_profit','cast_6_profit','cast_7_profit','cast_8_profit','cast_9_profit','cast_10_profit','others_1_profit','others_2_profit','others_3_profit','others_4_profit','others_5_profit','others_6_profit','others_7_profit','others_8_profit','others_9_profit','others_10_profit','prod_company_1_rev_budget_ratio','prod_company_2_rev_budget_ratio','prod_company_3_rev_budget_ratio','prod_company_4_rev_budget_ratio','prod_company_5_rev_budget_ratio','writer_1_rev_budget_ratio','writer_2_rev_budget_ratio','writer_3_rev_budget_ratio','writer_4_rev_budget_ratio','writer_5_rev_budget_ratio','director_1_rev_budget_ratio','director_2_rev_budget_ratio','director_3_rev_budget_ratio','cast_1_rev_budget_ratio','cast_2_rev_budget_ratio','cast_3_rev_budget_ratio','cast_4_rev_budget_ratio','cast_5_rev_budget_ratio','cast_6_rev_budget_ratio','cast_7_rev_budget_ratio','cast_8_rev_budget_ratio','cast_9_rev_budget_ratio','cast_10_rev_budget_ratio','others_1_rev_budget_ratio','others_2_rev_budget_ratio','others_3_rev_budget_ratio','others_4_rev_budget_ratio','others_5_rev_budget_ratio','others_6_rev_budget_ratio','others_7_rev_budget_ratio','others_8_rev_budget_ratio','others_9_rev_budget_ratio','others_10_rev_budget_ratio']]

	labels.to_csv('ml_data/labels.csv')
	input_features.to_csv('ml_data/input_features.csv')


# preProcessData()

# Data Setup
# Read Data, Drop index col (0)
# Select label
# 1 -- Vote Average
# 2 -- Revenue 
# 3 -- Rev Budget Ratio

label = 3

labels = np.loadtxt('ML/ml_data/labels.csv',skiprows=1,delimiter=',')[:,label] 
input_features = np.loadtxt('ML/ml_data/input_features.csv',skiprows=1,delimiter=',')[:,1:]
num_data 	   = labels.shape[0]


# Shuffle Data
shuffle_idx    = np.random.choice(num_data,size=num_data)
labels 		   = labels[shuffle_idx]
input_features = input_features[shuffle_idx]


# Normalise Labels
rev_mean = np.mean(labels[:3000])
rev_std  = np.std(labels[:3000])
labels = (labels-rev_mean)/rev_std


# Preprae Training and Test Data
train_labels = labels[:3000]
train_feat   = input_features[:3000]

test_labels = labels[3000:]
test_feat   = input_features[3000:]


'''
Random Forest Method
'''

rf = RandomForestRegressor(n_estimators = 100, random_state = 17,min_samples_split=2,min_samples_leaf=1)
rf.fit(train_feat,train_labels)
preds = rf.predict(test_feat)
errors = abs(preds - test_labels)
# print(np.mean(abs(np.exp(rev_mean+(preds*rev_std)) - np.exp(rev_mean+(test_labels*rev_std)))))


# if(label==3):
# 	pickle.dump(rf, open('models/revenue_model_rf.p', 'wb'))
# elif(label==1):
# 	pickle.dump(rf, open('models/rating_model_rf.p', 'wb'))

# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# # Print the feature ranking
# print("Feature ranking:")
# for f in range(test_feat.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(test_feat.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(test_feat.shape[1]), indices)
# plt.xlim([-1, test_feat.shape[1]])
# plt.show()


# '''
# Neural Network Method
# '''

# Model Definitions

# # Training Revenue or Revenue-to-Budget Ratio
# rev_model = Sequential()
# rev_model.add(Dropout(0.1,input_shape=(train_feat.shape[1],)))
# rev_model.add(Dense(1024,activation='relu'))
# rev_model.add(Dropout(0.2))
# rev_model.add(Dense(1024,activation='relu'))
# rev_model.add(Dropout(0.4))
# rev_model.add(Dense(1))
# rev_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
# rev_model.summary()


# # Vote Average/Rating
# rating_model = Sequential()
# rating_model.add(Dropout(0.1,input_shape=(train_feat.shape[1],)))
# rating_model.add(Dense(1024,activation='relu'))
# rating_model.add(Dropout(0.2))
# rating_model.add(Dense(1024,activation='relu'))
# rating_model.add(Dropout(0.4))
# rating_model.add(Dense(1,activation='sigmoid'))
# rating_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
# rating_model.summary()


# # Training
# if(label==1):
# 	rating_model.fit(train_feat, train_labels,
# 					validation_data=(test_feat,test_labels),
# 					epochs=250,
# 					batch_size=32,
# 					callbacks = [ModelCheckpoint('models/rating_model.h5', save_best_only=True, monitor='val_loss', mode='min')])
# elif(label==2 or label==3):
# 	rev_model.fit(train_feat,train_labels,
# 					validation_data=(test_feat,test_labels),
# 					epochs=250,
# 					batch_size=32,
# 					callbacks = [ModelCheckpoint('models/revenue_model.h5', save_best_only=True, monitor='val_loss', mode='min')])








