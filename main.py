# clustering mobile network operator's data using SOM

import pandas as pd
from sklearn_som.som import SOM
import matplotlib.pyplot as plt

# read and cleaning data

data = pd.read_csv('data/SELL.csv', )

data = data.drop(
    ['RESERVE_ID', 'TRANSACTION_CODE', 'REFID', 'ROUTE', 'SELL_PRICE', 'CUSTOMER_ID', 'REQUEST_ID', 'IS_NEW',
     'OPERATOR_REQUEST_ID', 'BENEFIT_PERCENT', 'SYSTEM_BENEFIT', 'RESELLER_BENEFIT', 'UPDATE_TIMESTAMP',
     'CHECK_STATUS_COUNT', 'CHANNEL_BALANCE_AFTER'], axis=1)

data['Dates'] = pd.to_datetime(data['SELL_TIMESTAMP']).dt.date
data['Dates'] = data['Dates'].astype("datetime64")
data['Time'] = pd.to_datetime(data['SELL_TIMESTAMP'], errors='coerce')
data['day_of_week'] = data['Dates'].dt.dayofweek
data['year'] = data["Dates"].dt.year
data['month'] = data["Dates"].dt.month
data['hour'] = data['Time'].dt.hour


data = data.drop(['SELL_TIMESTAMP', 'Dates', 'Time'], axis=1)

###############################################################################3

df = data.to_numpy()

# Build clusters and fit it to the data
som = SOM(m=3, n=1, dim=11)
som.fit(df)

predictions = som.predict(df)

##############################################################

# sample plots

#1
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')

plt.title("Plot 1")
ax.set_xlabel('Category ID',fontsize=10)
ax.set_ylabel('Buy Price',fontsize=10)
ax.set_zlabel('Vendor ID',fontsize=10)
ax.scatter3D(df[:, 2], df[:, 1], df[:, 6], c=predictions)

#2
ax = fig.add_subplot(2, 2, 2, projection='3d')

plt.title("Plot 2")
ax.set_xlabel('Category ID',fontsize=10)
ax.set_ylabel('Vendor ID',fontsize=10)
ax.set_zlabel('Month',fontsize=10)
ax.scatter3D(df[:, 2], df[:, 1], df[:, 9], c=predictions)

#3
ax = fig.add_subplot(2, 2, 3, projection='3d')

plt.title("Plot 3")
ax.set_xlabel('Channel ID',fontsize=10)
ax.set_ylabel('Buy Price',fontsize=10)
ax.set_zlabel('Year',fontsize=10)
ax.scatter3D(df[:, 3], df[:, 1], df[:, 8], c=predictions)

#4
ax = fig.add_subplot(2, 2, 4, projection='3d')

plt.title("Plot 4")
ax.set_xlabel('Product ID',fontsize=10)
ax.set_ylabel('Vendor ID',fontsize=10)
ax.set_zlabel('Day of week',fontsize=10)
ax.scatter3D(df[:, 5], df[:, 6], df[:, 7], c=predictions)

###############################

fig.tight_layout(pad=5.0)
plt.savefig('output.png')
plt.show()



# what is channel id?
