import keras
import numpy as np
import pandas as pd
import pickle

from keras.models import load_model
import matplotlib.pyplot as plt

np.random.seed(seed=2)

# Read Data, Drop index col (0)
# 1 -- Vote Average
# 2 -- Revenue 
# 3 -- Rev Budget Ratio

label  		   = 3
labels 		   = np.loadtxt('ml_data/labels.csv',skiprows=1,delimiter=',')[:,label] 
input_features = np.loadtxt('ml_data/input_features.csv',skiprows=1,delimiter=',')[:,1:]

# Load Model
if(label == 1):
	model = load_model('models/rating_model.h5')
	model.summary()
elif(label==2 or label==3):
	# model = pickle.load(open('models/revenue_model_rf.p', 'rb'))
	model = load_model('models/revenue_model.h5')
	model.summary()


# Normalize Labels
mean   = np.mean(labels[:3000])
std    = np.std(labels[:3000])
labels = (labels-mean)/std

num_data 	   = labels.shape[0]
shuffle_idx    = np.random.choice(num_data,size=num_data)
labels 		   = labels[shuffle_idx]
input_features = input_features[shuffle_idx]

train_labels = labels[:3000]
train_feat   = input_features[:3000]

test_labels = labels[3000:]
test_feat   = input_features[3000:]

out 		= model.predict(test_feat)

# Use this for NN
rev_pred = (((out*std)+mean)[:,0])

# Use this for RF
# rev_pred = (((out*std)+mean))

rev_test = ((test_labels*std)+mean)
sort_idx = np.argsort(rev_test)




num_test_movies = rev_pred.shape[0]
plt.scatter(np.arange(num_test_movies),rev_pred[sort_idx],c='blue')
plt.scatter(np.arange(num_test_movies),rev_test[sort_idx],c='green')
plt.bar(np.arange(num_test_movies),abs(rev_pred[sort_idx]-rev_test[sort_idx]),color='red')
plt.show()