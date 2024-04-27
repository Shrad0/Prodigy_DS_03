# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:51:54 2024

@author: Shraddha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

df = pd.read_csv('E:\Projects/Decision Tree/bank+marketing/bank-additional/bank-additional/bank-additional-full.csv', delimiter = ';')

df.rename(columns={'y':'deposit'}, inplace = 'True')

df.head()

df.tail()

df.shape

df.columns

df.dtypes

df.dtypes.value_counts()

df.info()

df.duplicated().sum()

df.isna().sum()

cat_cols = df.select_dtypes(include = 'object').columns
print(cat_cols)

num_cols = df.select_dtypes(exclude = 'object').columns
print(num_cols)

df.describe()

df.describe(include = 'object')

#Histogram
#df.hist(figsize = (0,10), color = 'red')
#plt.show()

#Bar Plot
#for feature in cat_cols:
 #   plt.figure()
  #  sns.countplot(x=feature, data=df, palette='Wistia')
  #  plt.title(f'Bar Plot of {feature}')
  #  plt.xlabel(feature)
  #  plt.ylabel('Count')
  #  plt.xticks(rotation=90)
  #  plt.show()
    
#Box Plot
df.plot(kind = 'box', subplots = True, layout = (2,5), color='#7b3f00')
plt.show()

column = df[['age','campaign','duration']]
q1 = np.percentile(column, 25)
q3 = np.percentile(column, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df[['age','campaign','duration']] = column[(column > lower_bound) & (column < upper_bound)]

df.plot(kind = 'box', subplots = True, layout = (2,5), color='#808000')
plt.show()


#non_numeric_columns = df.select_dtypes(exclude=['number']).columns

#for column in non_numeric_columns:
#    print(column, df[column].unique())

#df[housemaid] = df[numeric_colummns].apply(pd.to_numeric, errors = 'coerce')
#corr_matrix=df.corr()
#print(corr_matrix)


corr = df.corr()
print(corr)
corr = corr[abs(corr)>=0.90]
sns.heatmap(corr, annot = True, cmap = 'Set3', linewidths = 0.2)
plt.show()

#numeric_df = df.select_dtypes(include=['float64', 'int64'])

high_corr_cols = ['emp.var.rate', 'euribor3m', 'nr.employed']

df1 = df.copy()
df1.columns
