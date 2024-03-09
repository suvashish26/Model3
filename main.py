#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[1]:


import seaborn as sns
import numpy as np


# In[3]:


import pandas as pd


# In[8]:


import pandas as pd
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the data directory
data_dir = os.path.join(script_dir, '..', 'data')

# Define the file name
file_name = 'kalimati.csv'

# Construct the full path to the CSV file
file_path = os.path.join(data_dir, file_name)

# Read the CSV file
df = pd.read_csv(file_path, na_values="=")



# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


data2= df.copy()


# In[12]:


data2.head()


# In[16]:


data2["Commodity"][1]


# In[51]:


str=data2['Date'][1]
str2= str.split('/')
print(str)
print(str2)
print(str2[1])


# In[52]:


Dict= {1:"January",2:"February",3:"March", 4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
print(Dict)


# In[53]:


Dict[3]


# In[54]:


month_column=[]


# In[62]:


# Assuming you have initialized month_column and Dict before this code snippet

for rr in data2['Date']:
    try:
        date_parts = rr.split('/')
        if len(date_parts) == 3:  # Assuming the date format is MM/DD/YYYY
            month_column.append(Dict[int(date_parts[0])])
        else:
            # Handle the case where the date format is not as expected
            month_column.append(None)
    except IndexError:
        # Handle the case where the split result does not have the expected number of elements
        month_column.append(None)


# In[63]:


len(month_column)


# In[64]:


data2['month_column']=month_column


# In[65]:


data2["month_column"]


# In[66]:


data2['month_column'].unique()


# In[67]:


season_names=[]


# In[68]:


for tt in data2["month_column"]:
    if tt == "January" or tt == "February":
        # print("winter")
        season_names.append("winter")
    elif tt == "March" or tt == "April":
        # print("spring")
        season_names.append("spring")
    elif tt == "May" or tt == "June":
        # print("summer")
        season_names.append("summer")
    elif tt == "July" or tt == "August":
        # print("monsoon")
        season_names.append("monsoon")
    elif tt == "September" or tt == "October":
        # print("autumn")
        season_names.append("autumn")
    elif tt == "November" or tt == "December":
        # print("pre winter")
        season_names.append("pre winter")


# In[69]:


data2["season_names"]=season_names


# In[70]:


data2.head()


# In[73]:


import pandas as pd
df= pd.Timestamp('2013/06/16')
print(df.dayofweek)


# In[74]:


day_of_week=[]


# In[75]:


for rr in data2['Date']:
    str= rr
    df= pd.Timestamp(rr)
    day= df.dayofweek
    day_of_week.append(day)
    
    


# In[76]:


len(day_of_week)


# In[77]:


data2['day']=day_of_week


# In[ ]:





# In[78]:


data2.tail()


# In[79]:


data2 = data2.drop('Date', axis=1)


# In[80]:


data2.columns


# In[82]:


import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib to show the plot

# Assuming data2 is your DataFrame
sns.boxplot(x=data2['Average'])
plt.xlabel('Average')  # Adding a label for the x-axis
plt.show()


# In[83]:


# IQR
import numpy as np
Q1 = np.percentile (data2 [ 'Average'], 25,interpolation = "midpoint")
Q3 = np.percentile (data2 ['Average'], 75, interpolation = 'midpoint')
IQR = Q3-Q1


# In[84]:


# Upper bound
upper = np.where(data2 [ 'Average'] >= (Q3+1.5*IQR))
#Lower bound
lower = np.where (data2 [ 'Average'] <= (Q1-1.5*IQR))


# In[85]:


print (upper [0], lower [0])


# In[86]:


data2.drop (upper [0], inplace = True)
data2.drop(lower [0], inplace = True)
print("New Shape: ", data2.shape)


# In[90]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=data2["Average"], orient="h")  # Set orient to "h" for horizontal orientation
plt.xlabel('Average')  # Adding a label for the x-axis
plt.show()


# In[91]:


df= data2.copy()


# In[92]:


data2.columns


# In[93]:


import plotly.express as px


# In[94]:


sns.relplot(data=df, y ="Average", x="season_names", hue= "season_names", kind="line")


# In[95]:


fig= px.bar(df, y ="Average", x="season_names",height = 400)
fig.show()


# In[96]:


data2.info()


# In[97]:


dist=(data2 ['Commodity'])
distset=set (dist)
dd= list(distset)
dictOfWords = { dd[i] : i for i in range(0, len (dd)) }
data2['Commodity']=data2 ['Commodity'].map(dictOfWords)


# In[98]:


dist=(data2 ['month_column'])
distset=set (dist)
dd= list(distset)
dictOfWords = { dd[i] : i for i in range(0, len (dd)) }
data2['month_column']=data2 ['month_column'].map(dictOfWords)


# In[99]:


dist=(data2 ['season_names'])
distset=set (dist)
dd= list(distset)
dictOfWords = { dd[i] : i for i in range(0, len (dd)) }
data2['season_names']=data2 ['season_names'].map(dictOfWords)


# In[100]:


data2.info()


# In[ ]:





# In[ ]:





# In[101]:


import matplotlib.pyplot as plt


# In[102]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'date_column' is a non-numeric column you want to exclude
numeric_columns = data2.select_dtypes(include=['float64', 'int64']).columns
numeric_data = data2[numeric_columns]

dataplot = sns.heatmap(numeric_data.corr(), cmap='YlGnBu', annot=True)
plt.show()


# In[103]:


data2.columns


# In[105]:


features = data2[['Commodity', 'month_column', 'season_names', 'day']]
labels= data2['Average']


# In[106]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels , test_size =0.2, random_state =2)


# In[107]:


import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[108]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr=RandomForestRegressor (max_depth=1000, random_state=0)
regr.fit (Xtrain, Ytrain)


# In[109]:


Xtest[0:1]


# In[110]:


y_pred=regr.predict(Xtest)


# In[111]:


from sklearn.metrics import r2_score


# In[112]:


r2_score(Ytest, y_pred)


# In[113]:


y_pred


# In[114]:


data2.columns


# In[118]:


# Example: Use the second row in Xtest
user_input_alternative = Xtest.iloc[1:2]
prediction_alternative = regr.predict(user_input_alternative)
print(f'Alternative prediction: {prediction_alternative}')



# In[121]:


user_input = [[16, 3, 1, 3]]  # Example: Change season_names and day
prediction = regr.predict(user_input)
print(f'Updated prediction: {prediction}')


# In[ ]:




