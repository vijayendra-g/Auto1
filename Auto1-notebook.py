#!/usr/bin/env python
# coding: utf-8

# ##  Auto1 data science challenge
# #### Question 1 (10 Points) List as many use cases for the dataset as possible.
# 
# 1. **Price recommendation:** Auto1 values and buys used cars from individuals, dealerships, and manufacturers, then sells them          for a profit to other dealerships/ Customers. **Price recommendation** feature will help Auto1 to determine, what must be right value/price of car based on car features. We can train machine learning model using given data, to help Auto1 predict **right price** for test/unseen data depending on features of the car.
# 2. **Similar car recommendation:** Customers often want to see and compare similar cars. Using given data, we can recommend            **similar cars **  to the query/given car.  This will enhance user experience and user engagement.
# 3. **Predict Fuel-efficiency:**  Predict **avg-mpg** where **avg-mpg** is the average of **city and highway mileage**. As a customer, When I am buying a used car, I want to be sure of car's health/efficiency apart from other things. **avg-mpg** is usually overstated by the car seller and must be believed with a pinch of salt.
#      
#      As a customer I would like **Auto1** run machine learning models and estimate this parameter(**avg-mpg**) for a used car.   **avg-mpg**  is also  one  of the indicators of healthy car.
#      
#      As a customer I want to know what my  car **avg-mpg** is going to be in a world where  petrol/disel rates go up everyday. It would be great help for the user         if we can predict the **avg-mpg** using  features of cars.
# 4. **Symboling:** Predict symboling ***[ 3,  1,  2,  0, -1, -2]***  based on car feature.

# #### By: Vijayendra

# ## 1. Import statements & Data Understanding

# In[1]:


#import statements
import pandas as pd
import numpy as np
import math
import scipy
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import feature_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataDF = pd.read_csv("Auto1-DS-TestData.csv", sep = ",")
print("shape of dataDF {0}\n\n".format(dataDF.shape))
dataDF.info()


#  * There are in all 205 records,26 features.
#  * missing values are represented as(?) as opposed to NaN 
#  * Some features such as price, normalized-losses, peak-rpm, horsepower, bore and stroke are represented as categorial because     of missing values(?).

# In[3]:


dataDF['symboling'].unique()


# ## 2. preprocessing
#      2.1 Handling Missing Values
#      2.2 Outlier detection

# ###  2.1 Handling missing values

# In[4]:


dataDF.isin(['?']).sum(axis=0)


# Missing values in various columns.
#       1. normalized-losses  :  41
#       2. num-of-doors      :    2
#       3. bore               :   4
#       4. stroke             :   4
#       5. horsepower          :  2
#       6. peak-rpm            :  2
#       7. price               :  4
#       
# There are various stategies to handle missing values. As there are only 206 records, we will not throway missing records, instead **carefully  Intelligently** inpute them.

# #### Impute num-of-doors feature

# In[5]:


dataDF[dataDF['num-of-doors'] =='?']


# In[6]:


print(dataDF[dataDF['body-style']=='sedan']['num-of-doors'].value_counts())


# In[7]:


dataDF['num-of-doors']=dataDF['num-of-doors'].map({'two':2,'four':4,'?':4})


# * Looks like majority of sedans have four doors. We will impute 4 for the missing values.
# * We will also **map({'two':2,'four':4,'?':4}**  i.e convert the **num-of-doors** to numeric feature.

# ### Impute  horsepower

# * We have talked about **intelligent imputing**. I will describe it here. Let us do it for **horsepower**.
# * **simple method Imputation** : take the mean and impute it. in this approach the mean would be 104.25.
# * **Intellegent method Imputation** : Subset the data on different features based on your knowledge of features and impute value   you get.  This method gives you better estimate of imputation. 
# 
#   Ex: We have subsetted the data based on  **dataDF[  (dataDF['fuel-type']=='gas') & (dataDF['body-style'] == 'hatchback') & (dataDF['num-of-cylinders'] == 'four') & (dataDF['horsepower']!='?')]['horsepower'].mean()**
#   
#   The value imputed is **92.04**

# In[8]:


dataDF[dataDF['horsepower']=='?']


# In[9]:


# Simple method Imputation

tempDF=dataDF[dataDF['horsepower']!='?']
hp_mean=(tempDF['horsepower'].astype(int)).mean()
print(hp_mean)


# In[10]:


hp_mean  = dataDF[  (dataDF['fuel-type']=='gas') & (dataDF['body-style'] == 'hatchback') & (dataDF['num-of-cylinders'] == 'four') & (dataDF['horsepower']!='?')]['horsepower'].astype(int).mean()
dataDF['horsepower']=dataDF['horsepower'].replace('?',hp_mean).astype(int)


# ##### For remaining imptations, I am using mean of the column  due to lack of time.
# ##### But a good inpute stratergy  always help us get good model

# ### Impute  bore feature

# In[11]:


tempDF=dataDF[dataDF['bore']!='?']
bore_mean=(tempDF['bore'].astype(float)).mean()
dataDF['bore']=dataDF['bore'].replace('?',bore_mean).astype(float)


# ### Impute stroke column

# In[12]:


tempDF=dataDF[dataDF['stroke']!='?']
stroke_mean=(tempDF['stroke'].astype(float)).mean()
dataDF['stroke']=dataDF['stroke'].replace('?',stroke_mean).astype(float)


# ### Impute peak-rpm

# In[13]:


tempDF=dataDF[dataDF['peak-rpm']!='?']
rpm_mean=(tempDF['peak-rpm'].astype(float)).mean()
dataDF['peak-rpm']=dataDF['peak-rpm'].replace('?',rpm_mean).astype(float)


# ### Impute  number-of-cylinders

# In[14]:


dataDF['num-of-cylinders'].unique()


# In[15]:


dataDF['num-of-cylinders']=dataDF['num-of-cylinders'].map({'three':3,'four':4,'five':5,'six':6,'?':4, 'twelve':12, 'two':2, 'eight':8})


# In[16]:


dataDF.isnull().sum()


# In[17]:


dataDF.isin(['?']).sum(axis=0)


# ### Impute normalized-losses

# In[18]:


tempDF=dataDF[dataDF['normalized-losses']!='?']
nor_mean=(tempDF['normalized-losses'].astype(int)).mean()
dataDF['normalized-losses']=dataDF['normalized-losses'].replace('?',nor_mean).astype(int)


# ### Impute price

# In[19]:


tempDF = dataDF[dataDF['price']!='?']
price_avg=(tempDF['price'].astype(int)).mean()
dataDF['price']=dataDF['price'].replace('?',price_avg).astype(int)


# In[20]:


dataDF.head()


# **Now we are good to go, no missing values in dataframe.**

# * Basic statistics of dataDf

# In[21]:


dataDF.describe()


# ### Question 2 (10 Points)
# ###### Auto1 has a similar dataset (yet much larger...)  Pick one of the use cases you listed in question 1 and describe how building a statistical model based on the dataset could best be used to improve Auto1’s business.
# 

# ### Predict Fuel-efficiency: 
# 
# **Predict Fuel-efficiency:**  Predict **avg-mpg** where **avg-mpg** is the average of **city and highway mileage**. As a customer, When I am buying a used car, I want to be sure of car's health/efficiency apart from other things. **avg-mpg** is usually overstated by the car seller and must be believed with a pinch of salt.
#      
# As a customer I would like **Auto1** run machine learning models and estimate this parameter(**avg-mpg**) for a used car.   **avg-mpg**  is also  one  of the indicators of healthy car.
#      
# As a customer I want to know what my  car **avg-mpg** is going to be in a world where  petrol/disel rates go up everyday. It would be great help for the user         if we can predict the **avg-mpg** using  features of cars.
# 
#     1. EDA : What features correlate/affect MPG?
#     2. Methods:
#         2.1 Tree based methods for predicting **avg-mpg**
#         2.2 Linear model
# 

# We will create one variable called **avg-mpg**

# In[22]:


dataDF['avg-mpg'] = (dataDF['city-mpg']+dataDF['highway-mpg'])/2
dataDF.drop(['city-mpg','highway-mpg'],axis=1,inplace=True)


# ## 3. EDA

# ####  Correlation avg-mph with other numeric variables

# In[23]:


corr = dataDF.corr()
corr_map = sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)


# In[24]:


corr['avg-mpg']


# * **avg-mpg** positively correlates with compression-ratio
# * **horsepower, curb-weight,engine-size** negatively correlates

# In[25]:


plot_color = "#55AA33"
palette = ["#FFA07A","#FF0000", "#dd0033","#800000","#DB7093"]
figalpha = 0.5
axalpha = 0

left   =  0.10  # the left side of the subplots of the figure
right  =  0.95    # the right side of the subplots of the figure
bottom =  0.2    # the bottom of the subplots of the figure
top    =  0.3    # the top of the subplots of the figure
wspace =  0.1     # the amount of width reserved for blank space between subplots
hspace = 0.1 # the amount of height reserved for white space between subplots
y_title_margin = 1.0 # The amount of space above titles


# In[26]:


## undestanding few features

dataDF[['engine-size','peak-rpm','curb-weight','horsepower','price','avg-mpg']] .hist(figsize=(10,8),bins=8,color='g',linewidth='1',edgecolor='k')
plt.tight_layout()
plt.show()


# In[27]:



fig, ax = plt.subplots(figsize=(6,5), ncols=1, nrows=1) 
fig.patch.set_alpha(0.5)
ax.set_title("Horsepower - avg-mpg", y = y_title_margin, fontsize=16)
ax.patch.set_alpha(0)
gax8=sns.regplot("horsepower",'avg-mpg', data=dataDF, color=plot_color)
gax8.set_ylabel('avg-mpg',fontsize=14 )
gax8.set_xlabel('Horsepower',fontsize=14)



fig, ax = plt.subplots(figsize=(6,5), ncols=1, nrows=1)
fig.patch.set_alpha(0.5)
ax.set_title("engine-size - avg-mpg", y = y_title_margin, fontsize=16)
ax.patch.set_alpha(0)
gax8=sns.regplot("engine-size",'avg-mpg', data=dataDF, color=plot_color)
gax8.set_ylabel('avg-mpg',fontsize=14 )
gax8.set_xlabel('engine-size',fontsize=14)


# ### 4.1 Feature Engineering

# In[28]:


cat_vars = dataDF.select_dtypes(include = ["object"]).columns
num_vars = dataDF._get_numeric_data().columns


# In[29]:


print("Total no cat vars : {0} \nTotal no num vars : {1}\n".format(len(cat_vars),len(num_vars)))
print(cat_vars)
print(num_vars)


# We can code few cat variables as binary

# In[30]:


###  cat vars code
binary = ['fuel-type','aspiration','engine-location']
cat = ['make','body-style','engine-type','fuel-system','drive-wheels']

##  binary categories label encoder
binary_style = LabelBinarizer()

for var in binary:
    dataDF[var] = binary_style.fit_transform(dataDF[var])


# In[31]:


dataDF.shape


# In[32]:


dataDF.isnull().sum()


# In[33]:


## onehot encoding for categorical variables 
def onehotEncoding(df,features):

    for feature in features:
        dummies = pd.get_dummies(df[feature])
        df = pd.concat([df,dummies], axis=1) 
        df.drop([feature], axis=1, inplace=True)
    return df


# In[34]:


dataDF=onehotEncoding(dataDF,cat)


# In[35]:


dataDF.shape


# * In all we have 65 variables. 
# * We will make **avg-mpg** as response variable and **Remaining varaibles** as Predictors.

# In[36]:


# Helper function
def splitData(df, label):
    
    y = df[label]
    X = df.drop(label, axis = 1)
    return X, y


# In[37]:


X, y = splitData(dataDF, label = "avg-mpg")


# ### 4.2  Feature Selection

# In[38]:


# feature selection using cross validation
class PipelineRFE(Pipeline):

    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self

pipe = PipelineRFE(
    [
        ('std_scaler', preprocessing.StandardScaler()),
        ("ET", ExtraTreesRegressor(random_state=42, n_estimators=25))
    ]
)


# In[39]:


feature_selector_cv = feature_selection.RFECV(pipe, cv=5, step=1, scoring="neg_mean_squared_error")
feature_selector_cv.fit(X, y)


# In[40]:


print(feature_selector_cv.n_features_)
cv_grid_rmse = np.sqrt(-feature_selector_cv.grid_scores_)
print(cv_grid_rmse)


# * feature selection has given out 12 features. Lets print them 

# In[41]:


feature_names = (X.columns)
selected_features = feature_names[feature_selector_cv.support_].tolist()
print( selected_features)


# ## 5. Modeling

# In[42]:


X =  X[selected_features]


# In[43]:


# split data into train test
def trainingTesting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,random_state=10)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = trainingTesting(X, y)


# In[44]:


X_train.shape


# In[45]:


X_train.head()


# In[46]:


# helper function 
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[47]:


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor


# ### We will use  GridSearchCV to select hyper parameters using cross validation 

# In[48]:


rf_param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 30],
    'max_features': [3,4,5],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_rf = GridSearchCV(estimator = rf, param_grid = rf_param_grid,  scoring='neg_mean_squared_error',
                          cv = 3, n_jobs = -1, verbose = 2)
grid_rf.fit(X_train, y_train)
rfResultDF = pd.DataFrame(grid_rf.cv_results_)[['mean_test_score', 'params']]


# In[49]:


rfResultDF


# In[50]:


grid_rf.best_params_


# In[51]:


model_rf = RandomForestRegressor(n_estimators=200,max_depth = 10, max_features = 5) 
model_rf.fit(X_train, y_train)
rfTrainPred = model_rf.predict(X_train)
rfTrainError  = rmsle(y_train,rfTrainPred)
print("Random forest Training RMSE : {0}".format(rfTrainError))

rfTestPred = model_rf.predict(X_test)
rfTestError =  rmsle(y_test,rfTestPred)
print("Random forest Testing RMSE : {0}".format(rfTestError))
print("Random forest Testing R Squared : {0}".format( str(round(100 * r2_score(y_test, rfTestPred), 2)) + "%"))


# In[52]:


model_linear = linear_model.LinearRegression()
model_linear.fit(X_train, y_train)
linearTestPred = model_linear.predict(X_test)
print("Linear regression Testing R Squared : {0} ".format( str(round(100 * r2_score(y_test, linearTestPred), 2)) + "%"))
print("Linear regression Testing RMSE: {0}".format(str(round(mean_squared_error(y_test, linearTestPred), 2))))


# In[53]:


print( model_linear.coef_)


# In[ ]:





# ### Question 4 (60 Points)
#     A. Explain each and every of your design choices (e.g., preprocessing, model selection, hyper parameters, evaluation criteria). Compare and contrast your choices with alternative methodologies. 
# 
#     B. Describe how you would improve the model in Question 3 if you had more time.
# 

# ### B. Describe how you would improve the model in Question 3 if you had more time.
# 
# 

# * The model is already giving good accuracy close to 90%. We can further improve the model by 
#         1. Trying more complex model like XGboost, GBM, Neural nets
#         2. Ensembling many models and predicting the outcome would usually improve final result.
#         3. Adding more data. Try get more data from internet. We can generate new data using existing using SMOTE. 

# * I would love to work on car similarity recommendation. 
# * Brief Idea:
#            1. Represent  the car object in certain dimension(n-dimesion) depending on features.
#            2. Select a distance measure to find similar car to the query/given car.
#            3. As we have mixture of variables(numeric+categorical), we can make use of **GOWER Distance**
