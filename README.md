# Disaster_Building_Damage_Grade_Prediction

This is a competition hosted by **drivendata.com** which ends on 31st Dec 2019. 
This competition requires competitors to build to a machine learning model to predict level of damage cause to the building given their socio-economic-demographic information

The model is graded on an unseen test dataset using micro-averaged F-1 Score.   
**Reference:** [Click to open](https://www.drivendata.org/competitions/57/nepal-earthquake/page/135/)

**Dataset**  
There are 39 variables in the training dataset. The dataset is semi-anonymized by the competition host. building_id is the unique identifier of each record.

**Data Exploration**  
1.  There are no missing values
2.  The data is imbalanced, meaning there are not equal no. of records for each target variable.
3.  Around 10% of total data has buildings with age 0.Age is binned in the intervals of 5. So 0 age denotes building with age upto 4 years and so forth.
4.  There are few data points with age more that 100 years as well. 
5.  area_percentage and height_percentage filed is normalized.
6.  'r' type of foundation is the leading type of foundation found in most of the damaged buildings.

**Data Modelling**  
Below are few things that I tried to get maximum F-1 Score on test data. 

1.  Parameter tuning for Random Forest and XGBoost
2.  Upsampling and downsampling the training data to deal with class imbalance.

