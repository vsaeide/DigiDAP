import numpy as np
import pandas as pd
from som import SOM


tdf=pd.read_csv('data/sell2.csv', usecols= ['ID','BUY_PRICE','CATEGORY_ID','CHANNEL_ID','DATA','PRODUCT_ID','SELL_PRICE','VENDOR_ID'])
df=tdf[:10000]
df= df.to_numpy()

print(df.shape)


if __name__ == '__main__':


    som = SOM(10, 10)  # initialize a 10 by 10 SOM

    som.fit(df, 10000, save_e=True, interval=10)  # fit the SOM for 10000 epochs, save the error every 10 steps
    som.plot_error_history(filename='images/som_error.png')  # plot the training error history

    targets = np.array(5000 * [0] + 5000 * [1])  # create some dummy target values

    #visualize the learned representation with the class labels
    som.plot_point_map(df, targets, ['Class 0', 'Class 1','Class 2'], filename='images/som.png')
    som.plot_class_density(df, targets, t=0, name='Class 0', colormap='Greens', filename='images/class_0.png')
    som.plot_class_density(df, targets, t=0, name='Class 1', colormap='Greens', filename='images/class_1.png')
    som.plot_class_density(df, targets, t=0, name='Class 2', colormap='Greens', filename='images/class_2.png')

    som.plot_distance_map(colormap='Blues', filename='images/distance_map.png')  # plot the distance map after training

    # predicting the class of a new, unknown datapoint
    datapoint=tdf[0:1]
    datapoint=datapoint.to_numpy()
    print(datapoint)
    print("Labels of neighboring datapoints: ", som.get_neighbors(datapoint, df, targets, d=0))


