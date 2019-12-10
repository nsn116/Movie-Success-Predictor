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
num_data 	   = labels.shape[0]

# Load Model
if(label == 1):
	model = load_model('models/rating_model.h5')
	model.summary()
	rf_model = pickle.load(open('models/rating_model_rf.p', 'rb'))
elif(label==2 or label==3):
	model = load_model('models/revenue_model.h5')
	model.summary()
	rf_model = pickle.load(open('models/revenue_model_rf.p', 'rb'))

# Normalize Labels
mean   = np.mean(labels[:3000])
std    = np.std(labels[:3000])
labels = (labels-mean)/std

# Shuffle Data
shuffle_idx    = np.random.choice(num_data,size=num_data)
labels 		   = labels[shuffle_idx]
input_features = input_features[shuffle_idx]


# Prepare Training and Testing Data
train_labels = labels[:3000]
train_feat   = input_features[:3000]
test_labels  = labels[3000:]
test_feat    = input_features[3000:]

num_test_movies = test_labels.shape[0]

# Predict with NN
out_nn 		= model.predict(test_feat)
rev_pred_nn = (((out_nn*std)+mean)[:,0])


# Predict with RF
out_rf 		= rf_model.predict(test_feat)
rev_pred_rf = (((out_rf*std)+mean))


# Get ground truth
rev_test_gt = ((test_labels*std)+mean)


sort_idx = np.argsort(rev_test_gt)


fig_nn,ax_nn = plt.subplots()
fig_rf,ax_rf = plt.subplots()

# Plot for NN
ax_nn.scatter(np.arange(num_test_movies),rev_test_gt[sort_idx],c='blue',alpha=0.5)
ax_nn.scatter(np.arange(num_test_movies),rev_pred_nn[sort_idx],c='green',alpha=0.4)
ax_nn.set_xlabel('Movie Index')
ax_nn.set_ylabel('log rev-to-budget ratio')
ax_nn.set_title('Prediction of log rev-to-budget ratio using Neural Network')
# ax_nn.set_ylabel('Normalized Rating')
# ax_nn.set_title('Prediction of Normalized Rating with Neural Network')
ax_nn.set_xlim([-5,705])
ax_nn.set_ylim([-14,14])
ax_nn.legend(['Ground Truth', 'Prediction'])
ax_nn.axhline(y=0.0,color='r')
# plt.figure(fig_nn.number)

# Plot for RF
ax_rf.scatter(np.arange(num_test_movies),rev_test_gt[sort_idx],c='blue',alpha=0.5)
ax_rf.scatter(np.arange(num_test_movies),rev_pred_rf[sort_idx],c='green',alpha=0.4)
ax_rf.set_xlabel('Movie Index')
ax_rf.set_ylabel('log rev-to-budget ratio')
ax_rf.set_title('Prediction of log rev-to-budget ratio using Random Forest')
# ax_nn.set_ylabel('Normalized Rating')
# ax_rf.set_title('Prediction of Normalized Rating with Random Forest')
ax_rf.set_xlim([-5,705])
ax_rf.set_ylim([-14,14])
ax_rf.legend(['Ground Truth', 'Prediction'])
ax_rf.axhline(y=0.0,color='r')
# fig_rf.show()


plt.show()



