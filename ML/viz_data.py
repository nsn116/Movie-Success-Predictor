import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../tmp_encoded_data.csv')

rev 	= df['vote_average']
log_rev = np.log(df['vote_average'])
norm_rev = (log_rev-np.mean(log_rev))/np.std(log_rev)

plt.hist(norm_rev,bins=100)
plt.show()