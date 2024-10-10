import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_hub as


df = pd.read_csv( "Datasets\wine-reviews.CSV",usecols= ['country', 'description','points','price','variety','winery'])

print(df.head())

df = df.dropna(subset=['description','points'])
print(df.head())


plt.hist(df.points,bins=20)
plt.show()