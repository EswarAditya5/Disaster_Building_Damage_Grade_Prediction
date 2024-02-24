# importing required libraries
import pandas as pd
import numpy as np

df=pd.read_csv('train.csv')
test1=pd.read_csv('test.csv')

# We need to remove building_id in both train and test files because it's doesn't change prediction
dr_tr=dr_tr.drop(['building_id','geo_level_1_id','geo_level_2_id','geo_level_3_id'],axis=1)
dr_tt=dr_tt.drop(['building_id','geo_level_1_id','geo_level_2_id','geo_level_3_id'],axis=1)

# converting datatype in train file
int_columns = ['count_floors_pre_eq','has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick',
        'has_superstructure_timber','has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel','has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry','has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other','damage_grade']

df[int_columns] = df[int_columns].astype('object')

# converting datatypes in test file
int_columns_tt = ['count_floors_pre_eq','has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick',
        'has_superstructure_timber','has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel','has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry','has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other']

test1[int_columns_tt] = test1[int_columns_tt].astype('object')

# splitting into numcols and objcols
numcols=df.select_dtype(np.number)
objcols=df.select_dtype('np.number')

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
numcols_mm=mm.fit_transform(numcols)
numcols_mm=pd.DataFrame(numcols_mm,columns=numcols.columns)

# splitting into numcols and objcols test file
numcols1=test1.select_dtype(np.number)
objcols1=test1.select_dtype('np.number')

numcols_mm_tt=mm.fit_transform(numcols1)
numcols_mm_tt=pd.DataFrame(numcols_mm_tt,columns=numcols1.columns)

# label encoding the object variables
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
objcols=objcols.apply(LabelEncoder().fit_transform)

objcols1=objcols1.apply(LabelEncoder().fit_transform)

# combining the both numcols and objcols
combinedf=pd.concat([numcols,objcols],axis=1)

test1=pd.concat([numcols_tt_mm,objcols1],axis=1)

# splitting into dependent varibale(y) and independent variables(X's)
X=combinedf.drop('damage_grade',axis=1)
y=combinedf.damage_grade

# Decission Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth=50, criterion='entropy',splitter='random')
treemodel=tree.fit(X,y)
treemodel.predict(X,y)
cross_val_score(tree,X,y)
np.mean([0.52589101, 0.52278571, 0.51742857, 0.50321429, 0.39735714])
tree_X=treemodel.predict(X)
pd.crosstab(y,tree_X)
print(classification_report(y,tree_X))

from imblearn.over_sampling import SMOTE
sm=SMOTE()
X,y=sm.fit_resample(X,y)


# Decission Tree
tree=DecisionTreeClassifier(max_depth=50, criterion='entropy',splitter='random')
treemodel=tree.fit(X,y)
treemodel.predict(X,y)
cross_val_score(tree,X,y)
np.mean([0.52589101, 0.52278571, 0.51742857, 0.50321429, 0.39735714])
tree_X=treemodel.predict(X)
pd.crosstab(y,tree_X)
print(classification_report(y,tree_X))
treepredict_test=treemodel.predict(test1)
treepredict_tt=pd.DataFrame(treepredict_test)
treepredict_tt=treepredict_tt.replace({0:'Low Damage',1:'Medium Damage',2:'High Damage'})
treepredict_tt.value_counts()

import joblib
joblib.dump(tree,'tree_model_smote.sav')
