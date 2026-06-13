import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


#1. load the data
housing=pd.read_csv("housing.csv")

#2. creating  a stratified test set
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5]) 

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_data=housing.iloc[train_index] #we will work on this data
    strat_test_data=housing.iloc[test_index]   # set aside the test da
    
#we will work on copy of training data
housing=strat_train_data.copy()

# 3.seperate features and labels
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

print(housing,housing_labels)  

# 4.list numerical and categorial values
num_attrb=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attrb=["ocean_proximity"]

# 5 lets make the pipelines
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler" ,StandardScaler()),
])

#pipeline for cat_attrb
cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder()),
])
 
#construction of full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attrb),
    ("cat",cat_pipeline,cat_attrb)
])

#6.transform the data
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared[:5]) 

#7. Train the model

# linear regression model 
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
lin_rmse=root_mean_squared_error(housing_labels,lin_preds)
#print(f"the root mean squared value for linear regression is {lin_rmse}")
lin_rmses= -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" ,cv=10 )
print(pd.Series(lin_rmses).describe())

# decision tree regressor model 
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse=root_mean_squared_error(housing_labels,dec_preds)
dec_rmses= -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" ,cv=10 )
# print(f"the root mean squared value for decision tree regressor is {dec_rmse}")
print(pd.Series(dec_rmses).describe())

# Random Forest regressor model 
rand_forst_reg=RandomForestRegressor()
rand_forst_reg.fit(housing_prepared,housing_labels)
rand_forst_preds=rand_forst_reg.predict(housing_prepared)
rand_forst_rmse=root_mean_squared_error(housing_labels,rand_forst_preds)
#print(f"the root mean squared value for random forest regressor is {rand_forst_rmse}")
rand_forst_rmses= -cross_val_score(rand_forst_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" ,cv=10 )
print(pd.Series(rand_forst_rmses).describe())



