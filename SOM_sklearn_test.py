import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn_som.som import SOM

# Load iris data

tdf=pd.read_csv('data/sell2.csv', usecols= ['ID','BUY_PRICE','CATEGORY_ID','CHANNEL_ID','DATA','PRODUCT_ID','SELL_PRICE','VENDOR_ID'])
df=tdf[:1000000]
df= df.to_numpy()



# Build clusters
som = SOM(m=3, n=8, dim=8, random_state=1234)

# Fit it to the data
som.fit(df)

# Assign each datapoint to its predicted cluster
predictions = som.predict(df)


colors = ['red', 'green', 'blue']


for i in range(4):
    for j in range(4,8):
        if not i==j:
            plt.scatter(df[:,i], df[:,j],c=predictions,cmap=ListedColormap(colors))
            plt.xlabel('value is '+str(i))
            plt.ylabel('value is '+str(j))
            plt.savefig('prediction/prediction'+str(i)+"_"+str(j)+'.png')



